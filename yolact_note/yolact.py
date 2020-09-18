import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt
from typing import List
from collections import defaultdict

from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage

# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
torch.cuda.current_device()   #查看当前使用的GPU序号

# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules,  JIT–Just-In-Time Compiler准时制
use_jit = torch.cuda.device_count() <= 1   #如果GPU数量<=1,打开JIT,如果>1,'Multiple GPUs detected! Turning off JIT.'
if not use_jit:
    print('Multiple GPUs detected! Turning off JIT.')

ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn



class Concat(nn.Module):
    def __init__(self, nets, extra_params):
        super().__init__()

        self.nets = nn.ModuleList(nets)
        self.extra_params = extra_params
    
    def forward(self, x):
        # Concat each along the channel dimension, 顺着通道维度拼接,(N,C,H,W) 
        return torch.cat([net(x) for net in self.nets], dim=1, **self.extra_params)  #**extra_params函数实参调用,为字典



def make_net(in_channels, conf, include_last_relu=True):    #如 make_net(256, [(256, 3, {'padding': 1})], include_last_relu=False)
    """
    A helper function to take a config setting and turn it into a network.
    Used by protonet and extrahead. Returns (network, out_channels)
    利用config中的网络设置来实现protonet和extrahead网络,返回 (network, out_channels)
    """
    def make_layer(layer_cfg): #layer_cfg就是以下Possible patterns,元组tuple,如( 256, 3, {})
        nonlocal in_channels   #外部嵌套函数make_net内的变量in_channels
        
        # Possible patterns: 可能的模式 [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})]
        # ( 256, 3, {}) -> conv   卷积
        # ( 256,-2, {}) -> deconv   反卷积
        # (None,-2, {}) -> bilinear interpolate   线性内插
        # ('cat',[],{}) -> concat the subnetworks in the list  #连接列表[]中的子网络
        #
        # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
        # Whatever, it's too late now.
        if isinstance(layer_cfg[0], str):  #若Possible patterns元组第一个元素是字符串,赋给layer_name,并调用外部嵌套函数make_net来构造网络
            layer_name = layer_cfg[0]    #这是专门针对layer_cfg = ('cat',[],{})  的情况

            if layer_name == 'cat':
                nets = [make_net(in_channels, x) for x in layer_cfg[1]]
                layer = Concat([net[0] for net in nets], layer_cfg[2])
                num_channels = sum([net[1] for net in nets])
        else:  #如果不是字符串
            num_channels = layer_cfg[0] #通道数量,卷积核个数
            kernel_size = layer_cfg[1]  #卷积核大小

            if kernel_size > 0:  #卷积核大于0的话,构造正常的卷积层
                layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])  #**layer_cfg[2]函数实参调用,layer_cfg[2]是一个字典变量
            else:
                if num_channels is None: #卷积核小于0,并且num_channels是None,则代表要线性插值了
                    layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                else:  ##卷积核小于0,num_channels  不是  None,则代表逆卷积
                    layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
        
        in_channels = num_channels if num_channels is not None else in_channels  #num_channels不是None,则令in_channels=num_channels

        # Don't return a ReLU layer if we're doing an upsample. This probably doesn't affect anything
        # output-wise, but there's no need to go through a ReLU here.
        # Commented out for backwards compatibility with previous models
        # if num_channels is None:
        #     return [layer]
        # else:
        return [layer, nn.ReLU(inplace=True)]  #返回的是一个list

    # Use sum to concat together all the component layer lists   #按照conf构造网络的层  
    #关于sum([[1,2,3]],[0,0,0]),把[1,2,3]加到[0,0,0]后边,组成[0,0,0,1,2,3],注意sum中的方括号,[[]],[]
    #第二点发现,每个layer(卷积层)后面都跟着一个relu层,
    net = sum([make_layer(x) for x in conf], [])  #conf=[(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})]
    #[Conv2d(32, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), ReLU(inplace=True), Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), ReLU(inplace=True),
    if not include_last_relu:  #不包含ReLU层,
        net = net[:-1]

    return nn.Sequential(*(net)), in_channels   # *(net),函数实参调用,加()转化为tuple,net原来是list,  in_channels最后输出的通道数
    """
    Protonet:
    Sequential(
  (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (5): ReLU(inplace=True)
  (6): InterpolateModule()
  (7): ReLU(inplace=True)
  (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): ReLU(inplace=True)
  (10): Conv2d(256, 32, kernel_size=(1, 1), stride=(1, 1))
) 
   32
    """

prior_cache = defaultdict(lambda: None)   #prior_cache={}  字典里若没有对应的key,那么返回一个默认值，而不报错

class PredictionModule(nn.Module): #prediction head模块
    """
    此函数载入数据之后，即forward部分返回preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }！！！！！！！！！！
    bbox：边界框坐标，conf:类别置信度，mask：掩模系数，priors：先验的anchor对应的边界框的坐标

    The (c) prediction module adapted from DSSD:预测模块改编自DSSD
    https://arxiv.org/pdf/1701.06659.pdf

    Bottleneck模块的中间层实际上是一个 3*3 的卷积核,(DSSD的论文里是1*1),此代码修改了过来.yolact作者没有试1*1的效果
    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.   输入特征通道数(层数)
        - out_channels:  The output feature size (must be a multiple of 4).  输出特征通道数必须是4的倍数
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - [[[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]]]

        - scales:        A list of priorbox scales relative to this layer's convsize.  #[[24], [48], [96], [192], [384]]
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
                        若最后的特征图是30*30,原图是600*600,特征图是原图下采样20倍得到的,那么特征图上的一个像素对应原图中的20*20个像素.
                        20就是base_anchor,scale是相对base anchor的.
                        当anchor的scale为1时,预测框为20*20.当scale为0.5时,为10*10
        - parent:        If parent is a PredictionModule, this module will use all the layers
                         from parent instead of from this module.
    
    一个Anchor Box可以由: 边框的纵横比 aspect_ratios 和 边框的面积（尺度scale) 来定义，
    在一幅图像中，要检测的目标可能出现在图像的任意位置，并且目标可能是任意的大小和任意形状。
    使用CNN提取的Feature Map的点，来定位目标的位置。
    使用Anchor box的Scale来表示目标的大小
    使用Anchor box的Aspect Ratio来表示目标的形状
    yolact使用了多个特征图来anchor，对于每个特征图只有一个固定的scale.(像faster r-cnn是一个特征图,对应三个scale尺度)

    Anchor Box的生成是以CNN网络最后生成的Feature Map上的点为中心的（映射回原图的坐标），
    
    映射回原图的坐标！！！！！！！！！！！！！！！！！！！！！！！
    映射回原图的坐标！！！！！！！！！！！！！！！！！！！！！！！
    映射回原图的坐标！！！！！！！！！！！！！！！！！！！！！！！
    映射回原图的坐标！！！！！！！！！！！！！！！！！！！！！！！

    PredictionModule(
    (upfeature): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
    )
    (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (mask_layer): Conv2d(256, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    ) 

            pred = PredictionModule(256, 256,
                                    aspect_ratios = [[1, 0.5, 2]],
                                    scales        = [24],#一共5个scale
                                    parent        = None,
                                    index         = 0)

    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1], parent=None, index=0):
        super().__init__()

        self.num_classes = cfg.num_classes    #len(my_custom_dataset.class_names) + 1,
        self.mask_dim    = cfg.mask_dim # Defined by Yolact  #32   mask系数矩阵   n*32 检测到的目标数*32
        self.num_priors  = sum(len(x) for x in aspect_ratios)  #3,代表三个不同纵横比的候选框,priors代表先验预测框  
        self.parent      = [parent] # Don't include this in the state dict
        self.index       = index
        self.num_heads   = cfg.num_heads # Defined by Yolact  #5

        #False,prediction head输出掩摸系数.If true, this will give each prediction head its own prototypes. 
        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:  
            self.mask_dim = self.mask_dim // self.num_heads

        if cfg.mask_proto_prototypes_as_features:  #False  把proto作为特征层再给输入？？？？
            in_channels += self.mask_dim
        
        if parent is None:
            if cfg.extra_head_net is None:  #不是none, [(256, 3, {'padding': 1})] 256, 3*3卷积
                out_channels = in_channels
            else:
                self.upfeature, out_channels = make_net(in_channels, cfg.extra_head_net) #out_channels=256  按照配置构造网络，
    # (upfeature): Sequential(
    #     (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (1): ReLU(inplace=True)
    # )
            if cfg.use_prediction_module: #False, Whether or not to use the prediction module (c) from DSSD
                self.block = Bottleneck(out_channels, out_channels // 4)
                self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                self.bn = nn.BatchNorm2d(out_channels)

            self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                **cfg.head_layer_params) #num_priors=3，*cfg.head_layer_params={'kernel_size': 3, 'padding': 1}
            self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
            self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,    **cfg.head_layer_params)
            
            if cfg.use_mask_scoring: #False,Adds another branch to the network to predict Mask IoU.
                self.score_layer = nn.Conv2d(out_channels, self.num_priors, **cfg.head_layer_params)

            if cfg.use_instance_coeff: #False
                self.inst_layer = nn.Conv2d(out_channels, self.num_priors * cfg.num_instance_coeffs, **cfg.head_layer_params)
            
            # What is this ugly lambda doing in the middle of all this clean prediction module code?
            def make_extra(num_layers):
                if num_layers == 0:
                    return lambda x: x
                else:
                    # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                    return nn.Sequential(*sum([[
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True)
                    ] for _ in range(num_layers)], []))

            self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers] #'extra_layers': (0, 0, 0),
                # cfg.extra_layers   (0, 0, 0),在基础网络和heads之间加一个额外层，都是0，
                # 那么意思是，如果是0，那么就直接return  FPN层的输出值了，就是不加中间层的意思！！！
                #下边是在FPN层后的upfeature之后用这个，输出upfeature的convout数据，送给bbox_layer等三个层
                # Add extra layers between the backbone and the network heads
                # The order is (bbox, conf, mask)
            if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:  #False
                self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None
        self.last_img_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])
                 x是输入,FPN层的输出,torch.Size([1, 256, 69, 69]) 
                 torch.Size([1, 256, 35, 35]) 
                 torch.Size([1, 256, 18, 18]) 
                 torch.Size([1, 256, 9, 9]) 
                 torch.Size([1, 256, 5, 5]) 

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes 返回有尺寸的元组(边界框的坐标,类别置信度,掩摸输出,预测框???)
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]  #num_priors 3
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        # In case we want to use another module's layers
        src = self if self.parent[0] is None else self.parent[0] 
        #第一次循环,self.parent里什么都没有,[None],返回self;
        # 其余四次迭代,self.parent不为none,则返回self.parent[0]
        
        conv_h = x.size(2) #FPN层的输出的大小
        conv_w = x.size(3)
        
        if cfg.extra_head_net is not None: #[(256, 3, {'padding': 1})] 
            x = src.upfeature(x) #先对特征金字塔层的输出Pi的数据,进行一个3*3卷积,然后relu, 
                                 #之前已经定义了这个函数，在forward里边主要是带入数据，x，串成一个链
        
        if cfg.use_prediction_module:  #False
            # The two branches of PM design (c)
            a = src.block(x)
            
            b = src.conv(x)
            b = src.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox_x = src.bbox_extra(x) #bbox_extra本意也是建立一个网络,但是按照config的配置,这里就是返回原数据,把x赋给bbox_x,bbox_x = x
        conf_x = src.conf_extra(x) #上边有解释，这里的extra网络就是   直接输出   upfeature层的   输出数据
        mask_x = src.mask_extra(x)

    # (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        #bbox和class
        bbox = src.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4) #Pi在经过1个upfeature3*3卷积之后,开始进入分类和定位网络
        conf = src.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes) # permute对维度进行换位,view相当于numpy.reshape() 
    
""" 总和 19248
torch.Size([1, 14283, 4]) torch.Size([1, 14283, 81]) 69*69*3,   一共14283个候选框??? 这个4代表bbox中心坐标和长宽
torch.Size([1, 3675, 4]) torch.Size([1, 3675, 81]) 
torch.Size([1, 972, 4]) torch.Size([1, 972, 81]) 
torch.Size([1, 243, 4]) torch.Size([1, 243, 81]) 
torch.Size([1, 75, 4]) torch.Size([1, 75, 81])  
"""

        if cfg.eval_mask_branch:  #True  Eval.py sets this if you just want to run YOLACT as a detector
            mask = src.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)  #self.mask_dim = 32
        #mask输出，这里输出的只是 mask_dim （32）个系数，
        else:
            mask = torch.zeros(x.size(0), bbox.size(1), self.mask_dim, device=bbox.device)

        if cfg.use_mask_scoring: #False  #类似bbox的置信度，mask也可以有自己的置信度，但是此网络默认没有
            score = src.score_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 1)

        if cfg.use_instance_coeff:  #False 
            inst = src.inst_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, cfg.num_instance_coeffs)    

        # See box_utils.decode for an explanation of this
        if cfg.use_yolo_regressors:   #False
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.eval_mask_branch:  #True   Eval.py sets this if you just want to run YOLACT as a detector
            if cfg.mask_type == mask_type.direct:
                mask = torch.sigmoid(mask)
            elif cfg.mask_type == mask_type.lincomb:
                mask = cfg.mask_proto_coeff_activation(mask)  #用tanh来激活mask系数 !!!!

                if cfg.mask_proto_coeff_gate:  #False
                    gate = src.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)  #生成一组门限值来过滤mask系数
                    mask = mask * torch.sigmoid(gate)

        if cfg.mask_proto_split_prototypes_by_head and cfg.mask_type == mask_type.lincomb:  #False   If true, this will give each prediction head its own prototypes
            mask = F.pad(mask, (self.index * self.mask_dim, (self.num_heads - self.index - 1) * self.mask_dim), mode='constant', value=0)
        
        priors = self.make_priors(conv_h, conv_w, x.device) #返回根据anchor机制生成的预选框坐标

        preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }

        if cfg.use_mask_scoring:  #False
            preds['score'] = score

        if cfg.use_instance_coeff:  #False
            preds['inst'] = inst
        
        return preds
    
    def make_priors(self, conv_h, conv_w, device):  #这一部分是根据anchor机制生成预选框
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. 
        候选框(x,y)是中心点坐标"""
        global prior_cache
        size = (conv_h, conv_w)   #FPN输出特征图的h,w  

        with timer.env('makepriors'): #timer.env()自动计时
            #第一个循环self.last_img_size=None,循环末赋值self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)
            if self.last_img_size != (cfg._tmp_img_w, cfg._tmp_img_h): #self.last_img_size=None,
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)): #意思就是遍历输出特征图的每个点,(0,0)开始，一个点(x，y)对应三个scale的预选框，这一步之后预选框*550，变成了图像上的预选框
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for scale, ars in zip(self.scales, self.aspect_ratios): #zip的作用,把列表对应元素组合起来,用括号括起来:[(24,[1,0.5,2])]
                        for ar in ars:
                            if not cfg.backbone.preapply_sqrt:  #False
                                ar = sqrt(ar)

                            if cfg.backbone.use_pixel_scales: #True
                                w = scale * ar / cfg._tmp_img_w  # These are populated by    #cfg._tmp_img_w参数由Yolact.forward()填充,550
                                h = scale / ar / cfg._tmp_img_h  # Yolact.forward
                            else:
                                w = scale * ar / conv_w
                                h = scale / ar / conv_h
                            
                            # This is for backward compatability with a bug where I made everything square by accident
                            if cfg.backbone.use_square_anchors:  #True
                                h = w
"""
#这里的[x, y, w, h]大小是相对于特征图的，不是相对于输入图像550×550的
"""
                            prior_data += [x, y, w, h] 
                
                self.priors = torch.Tensor(prior_data, device=device).view(-1, 4).detach() #转成pytorch格式,并reshape
                self.priors.requires_grad = False
                self.last_img_size = (cfg._tmp_img_w, cfg._tmp_img_h)  #(550,550)
                self.last_conv_size = (conv_w, conv_h)                 #
                prior_cache[size] = None
            elif self.priors.device != device:
                # This whole weird situation is so that DataParalell doesn't copy the priors each iteration
                if prior_cache[size] is None:
                    prior_cache[size] = {}
                
                if device not in prior_cache[size]:
                    prior_cache[size][device] = self.priors.to(device)

                self.priors = prior_cache[size][device]
        
        return self.priors #返回的是[x, y, w, h]
        """
        #self.priors输出的预选框都是正方形的，长和宽一致。
        它这里的候选框是按比例算的，如果最后乘上原图的size，整个特征图对应的anchor的候选框，
        能把整张输入图遍历完，5×5特征图比较粗略anchor,69*69特征图细致anchor
        这个是5*5特征图对应的anchor,
        tensor([[0.1000, 0.1000, 0.6982, 0.6982], 
        [0.1000, 0.1000, 0.4937, 0.4937],
        [0.1000, 0.1000, 0.9874, 0.9874],
        [0.3000, 0.1000, 0.6982, 0.6982],
        [0.3000, 0.1000, 0.4937, 0.4937],
        [0.3000, 0.1000, 0.9874, 0.9874],
        [0.5000, 0.1000, 0.6982, 0.6982],
        [0.5000, 0.1000, 0.4937, 0.4937],
        [0.5000, 0.1000, 0.9874, 0.9874],
        [0.7000, 0.1000, 0.6982, 0.6982],
        [0.7000, 0.1000, 0.4937, 0.4937],
        [0.7000, 0.1000, 0.9874, 0.9874],
        [0.9000, 0.1000, 0.6982, 0.6982],
        [0.9000, 0.1000, 0.4937, 0.4937],
        [0.9000, 0.1000, 0.9874, 0.9874],
        [0.1000, 0.3000, 0.6982, 0.6982],
        [0.1000, 0.3000, 0.4937, 0.4937],
        [0.1000, 0.3000, 0.9874, 0.9874],
        [0.3000, 0.3000, 0.6982, 0.6982],
        [0.3000, 0.3000, 0.4937, 0.4937],
        [0.3000, 0.3000, 0.9874, 0.9874],
        [0.5000, 0.3000, 0.6982, 0.6982],
        [0.5000, 0.3000, 0.4937, 0.4937],
        [0.5000, 0.3000, 0.9874, 0.9874],
        [0.7000, 0.3000, 0.6982, 0.6982],
        [0.7000, 0.3000, 0.4937, 0.4937],
        [0.7000, 0.3000, 0.9874, 0.9874],
        [0.9000, 0.3000, 0.6982, 0.6982],
        [0.9000, 0.3000, 0.4937, 0.4937],
        [0.9000, 0.3000, 0.9874, 0.9874],
        [0.1000, 0.5000, 0.6982, 0.6982],
        [0.1000, 0.5000, 0.4937, 0.4937],
        [0.1000, 0.5000, 0.9874, 0.9874],
        [0.3000, 0.5000, 0.6982, 0.6982],
        [0.3000, 0.5000, 0.4937, 0.4937],
        [0.3000, 0.5000, 0.9874, 0.9874],
        [0.5000, 0.5000, 0.6982, 0.6982],
        [0.5000, 0.5000, 0.4937, 0.4937],
        [0.5000, 0.5000, 0.9874, 0.9874],
        [0.7000, 0.5000, 0.6982, 0.6982],
        [0.7000, 0.5000, 0.4937, 0.4937],
        [0.7000, 0.5000, 0.9874, 0.9874],
        [0.9000, 0.5000, 0.6982, 0.6982],
        [0.9000, 0.5000, 0.4937, 0.4937],
        [0.9000, 0.5000, 0.9874, 0.9874],
        [0.1000, 0.7000, 0.6982, 0.6982],
        [0.1000, 0.7000, 0.4937, 0.4937],
        [0.1000, 0.7000, 0.9874, 0.9874],
        [0.3000, 0.7000, 0.6982, 0.6982],
        [0.3000, 0.7000, 0.4937, 0.4937],
        [0.3000, 0.7000, 0.9874, 0.9874],
        [0.5000, 0.7000, 0.6982, 0.6982],
        [0.5000, 0.7000, 0.4937, 0.4937],
        [0.5000, 0.7000, 0.9874, 0.9874],
        [0.7000, 0.7000, 0.6982, 0.6982],
        [0.7000, 0.7000, 0.4937, 0.4937],
        [0.7000, 0.7000, 0.9874, 0.9874],
        [0.9000, 0.7000, 0.6982, 0.6982],
        [0.9000, 0.7000, 0.4937, 0.4937],
        [0.9000, 0.7000, 0.9874, 0.9874],
        [0.1000, 0.9000, 0.6982, 0.6982],
        [0.1000, 0.9000, 0.4937, 0.4937],
        [0.1000, 0.9000, 0.9874, 0.9874],
        [0.3000, 0.9000, 0.6982, 0.6982],
        [0.3000, 0.9000, 0.4937, 0.4937],
        [0.3000, 0.9000, 0.9874, 0.9874],
        [0.5000, 0.9000, 0.6982, 0.6982],
        [0.5000, 0.9000, 0.4937, 0.4937],
        [0.5000, 0.9000, 0.9874, 0.9874],
        [0.7000, 0.9000, 0.6982, 0.6982],
        [0.7000, 0.9000, 0.4937, 0.4937],
        [0.7000, 0.9000, 0.9874, 0.9874],
        [0.9000, 0.9000, 0.6982, 0.6982],
        [0.9000, 0.9000, 0.4937, 0.4937],
        [0.9000, 0.9000, 0.9874, 0.9874]])
        """

class FPN(ScriptModuleWrapper):  #输入[512,1024,2048]
    """
    Implements a general version of the FPN introduced in
    https://arxiv.org/pdf/1612.03144.pdf

    Parameters (in cfg.fpn):
        - num_features (int): The number of output features in the fpn layers. FPN层的输出特征数量256
        - interpolation_mode (str): The mode to pass to F.interpolate.   内插模式 
        - num_downsample (int): The number of downsampled layers to add onto the selected layers.   降采样层的个数2,P6,p7层
                                These extra layers are downsampled from the last selected layer.    
    argument和parameter的区别:
    简略描述为：parameter=形参(formal parameter)， argument=实参(actual parameter)。
    Args:
        - in_channels (list): For each conv layer you supply in the forward pass,  [512,1024,2048]就是conv层的C3C4C
                              how many features will it have?
    """
    __constants__ = ['interpolation_mode', 'num_downsample', 'use_conv_downsample', 'relu_pred_layers', #固定参数,在config.py中
                     'lat_layers', 'pred_layers', 'downsample_layers', 'relu_downsample_layers']

    def __init__(self, in_channels):  #in_channels = [512,1024,2048]
        super().__init__()

        #从卷积层到特征金字塔的1*1卷积
        self.lat_layers  = nn.ModuleList([
            nn.Conv2d(x, cfg.fpn.num_features, kernel_size=1)  #num_features=256
            for x in reversed(in_channels)
        ])

        # This is here for backwards compatability 确认是1*1卷积后的3*3卷积
        padding = 1 if cfg.fpn.pad else 0    #cfg.fpn.pad=True,FPN填充padding=1
        self.pred_layers = nn.ModuleList([
            nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=padding)  
            for _ in in_channels    #像这个for循环中的_,就只是个占位符,没有用到,循环的作用主要是把这个nn.Conv2d()复制三遍,[nn.Conv2d(),nn.Conv2d(),nn.Conv2d()]
        ])

        if cfg.fpn.use_conv_downsample:   #就是P6P7  2个特征金字塔的降采样层
            self.downsample_layers = nn.ModuleList([
                nn.Conv2d(cfg.fpn.num_features, cfg.fpn.num_features, kernel_size=3, padding=1, stride=2)
                for _ in range(cfg.fpn.num_downsample)
            ])
        
        self.interpolation_mode     = cfg.fpn.interpolation_mode    #bilinear,插值方法
        self.num_downsample         = cfg.fpn.num_downsample        #2,降采样层数
        self.use_conv_downsample    = cfg.fpn.use_conv_downsample     #True,使用降采样层
        self.relu_downsample_layers = cfg.fpn.relu_downsample_layers  #False,给降采样层(P6P7)加relu
        self.relu_pred_layers       = cfg.fpn.relu_pred_layers    #True,给常规层(我的理解P3P4P5这三层)加relu

    @script_method_wrapper
    def forward(self, convouts:List[torch.Tensor]):
        """
                outs = [outs[i] for i in cfg.backbone.selected_layers] #[1,2,3] 对应的是C3,C4,C5
                outs = self.fpn(outs) 
        Args:
            - convouts (list): A list of convouts for the corresponding layers in in_channels. [512,1024,2048]输入通道中相应层的输出列表
            convouts[2].size(),convouts[1].size(),convouts[0].size()
            torch.Size([1, 2048, 18, 18]) torch.Size([1, 1024, 35, 35]) torch.Size([1, 512, 69, 69])
        Returns:
            - A list of FPN convouts in the same order as x with extra downsample layers if requested.
        """

        out = []
        x = torch.zeros(1, device=convouts[0].device) #tensor([0.]),device 可选参数, the desired device of returned tensor. convouts[0].device = cuda:0 
        for i in range(len(convouts)):
            out.append(x)
        
        # out = [tensor([0.]), tensor([0.]), tensor([0.])] 

        # For backward compatability, the conv layers are stored in reverse but the input and output is
        # given in the correct order. Thus, use j=-i-1 for the input and output and i for the conv layers.
        j = len(convouts)  #3
        for lat_layer in self.lat_layers:
            j -= 1

            if j < len(convouts) - 1:
                _, _, h, w = convouts[j].size()
                x = F.interpolate(x, size=(h, w), mode=self.interpolation_mode, align_corners=False)
            
            x = x + lat_layer(convouts[j])
            out[j] = x   #out[2]对应C5 1*1后的输出,out[1]对应C4 1*1后+ P5插值的输出,out[0]对应C3 1*1后+ P4插值的输出,在这之后,得到了3*3卷积之前的输出
        
        # This janky second loop is here because TorchScript.之所以第二个循环是因为TorchScript。
        j = len(convouts)
        for pred_layer in self.pred_layers: #pred_layers:3*3卷积层 P5P4P3, nn.Conv2d(256, 256, kernel_size=3, padding=1)*3
            j -= 1
            out[j] = pred_layer(out[j]) 

            if self.relu_pred_layers:
                F.relu(out[j], inplace=True) #3*3卷积完了之后,在relu,输出P5P4P3

        cur_idx = len(out) #len(out)=3

        # In the original paper, this takes care of P6, 添加两个降采样层,如果不用卷积来降采样的话,就用maxpooloing降采样
        if self.use_conv_downsample:
            for downsample_layer in self.downsample_layers:
                out.append(downsample_layer(out[-1]))
        else:
            for idx in range(self.num_downsample):
                # Note: this is an untested alternative to out.append(out[-1][:, :, ::2, ::2]). Thanks TorchScript.
                out.append(nn.functional.max_pool2d(out[-1], 1, stride=2))
        #len(out)=5
        if self.relu_downsample_layers:
            for idx in range(len(out) - cur_idx):
                out[idx] = F.relu(out[idx + cur_idx], inplace=False)

        return out #torch.Size([1, 256, 69, 69]) torch.Size([1, 256, 35, 35]) torch.Size([1, 256, 18, 18]) torch.Size([1, 256, 9, 9]) torch.Size([1, 1024, 5, 5]) 



class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by changing them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction. 用来做预测的卷积层的序号
        - #list(range(1, 4)),
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - #[[24], [48], [96], [192], [384]]
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule) 
        - #[[[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]], [[1, 0.5, 2]]]
    """

    def __init__(self):
        super().__init__()

        self.backbone = construct_backbone(cfg.backbone) #构建resnet101_backbone基础网络
        #self.backbone.channels = [256, 512, 1024, 2048]

        if cfg.freeze_bn:       #'freeze_bn': False, cfg.freeze_bn为True,则冻结基础网络中的BN层
            self.freeze_bn()

        """
        这一部分主要是在设置protonet网络,用于生成原型mask
        """
        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        # 在此处计算mask_dim并将其添加回config.py的网络配置中。 确保早点调用Yolact的构造函数！
        #mask_type有两种,direct,lincomb,本网络用的是lincomb, direct主要用于对比
        if cfg.mask_type == mask_type.direct:  
            cfg.mask_dim = cfg.mask_size**2    #mask_size 16
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:   #是否在protonet输入中添加额外的网格特征。False
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file)) #是的话就载入'data/grid.npy'网格文件,size:[numgrids, h, w]
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src  #protonet网络的输入层是哪一个?config中   'mask_proto_src': 0,
            
            if self.proto_src is None: in_channels = 3   #None的话就令protonet的输入通道数=3
            elif cfg.fpn is not None: in_channels = cfg.fpn.num_features  #FPN单层特征图所包含的通道数,256.  执行这一个
            else: in_channels = self.backbone.channels[self.proto_src]   #C2层  256
            in_channels += self.num_grids

            # The include_last_relu=false here is because we might want to change it to another function 这里protonet没有包含最后的ReLU,放在其它函数中了
            #'mask_proto_net': [(256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (256, 3, {'padding': 1}), (None, -2, {}), (256, 3, {'padding': 1}), (32, 1, {})],   #(num_features, kernel_size, **kwdargs)
            #输入256*69*69,经过3次256*256*3*3,p=1,s=1卷积之后,256*69*69,然后2倍上采样,256*138*138,接着1次256*256*3*3,p=1,s=1卷积,256*138*138,最后来一次32*256*1*1卷积,32*138*138
            self.proto_net, cfg.mask_dim = make_net(in_channels, cfg.mask_proto_net, include_last_relu=False)

            if cfg.mask_proto_bias: #是否包括一个与所有原型掩摸相对应的额外系数。False
                cfg.mask_dim += 1

        """
        这一部分主要是基础网络和fpn
        """
        self.selected_layers = cfg.backbone.selected_layers   #'selected_layers': list(range(1, 4)),   [1, 2, 3]
        src_channels = self.backbone.channels   #基础网络每个模块对应的输出通道数,[256, 512, 1024, 2048] 对应C2 C3 C4 C5

        if cfg.fpn is not None:
            # Some hacky rewiring to accomodate the FPN
            self.fpn = FPN([src_channels[i] for i in self.selected_layers])   
            #[512,1024,2048] #只执行了构造函数部分,没有执行forward,
            # 只是实例化一个对象self.fpn=FPN(),下边的self.fpn(outs)载入了数据,会自动执行forward
            self.selected_layers = list(range(len(self.selected_layers) + cfg.fpn.num_downsample))  
            #self.selected_layers是根据需要而变化的，[0, 1, 2, 3, 4]
            src_channels = [cfg.fpn.num_features] * len(self.selected_layers)  #[256, 256, 256, 256, 256]

        self.prediction_layers = nn.ModuleList()
        cfg.num_heads = len(self.selected_layers)  #5

        for idx, layer_idx in enumerate(self.selected_layers):    #self.selected_layers = [0, 1, 2, 3, 4]
            # If we're sharing prediction module weights, have every module's parent be the first one
            parent = None
            if cfg.share_prediction_module and idx > 0:  #'share_prediction_module': True,
                parent = self.prediction_layers[0]  #第一个循环idx=0,所以不会执行这句.

            pred = PredictionModule(src_channels[layer_idx], src_channels[layer_idx],
                                    aspect_ratios = cfg.backbone.pred_aspect_ratios[idx],
                                    scales        = cfg.backbone.pred_scales[idx],
                                    parent        = parent,
                                    index         = idx)

            self.prediction_layers.append(pred)
            """
            'pred_aspect_ratios': [ [[1, 1/2, 2]] ]*5,
            'pred_scales': [[24], [48], [96], [192], [384]],

            对循环中的每一层执行一个PredictionModule，返回以下信息，储存在self.prediction_layers数组
            (prediction_layers): ModuleList(
                (0): PredictionModule(
                (upfeature): Sequential(
                    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                    (1): ReLU(inplace=True)
                )
                (bbox_layer): Conv2d(256, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (conf_layer): Conv2d(256, 243, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (mask_layer): Conv2d(256, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                )
                (1): PredictionModule()
                (2): PredictionModule()
                (3): PredictionModule()
                (4): PredictionModule()
            )

            forward 部分载入数据之后返回preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
            bbox：边界框坐标，conf:类别置信度，mask：掩模系数，priors：先验的anchor对应的边界框的坐标
            """

        # Extra parameters for the extra losses
        if cfg.use_class_existence_loss:  #False
            # This comes from the smallest layer selected
            # Also note that cfg.num_classes includes background
            self.class_existence_fc = nn.Linear(src_channels[-1], cfg.num_classes - 1)
        
        if cfg.use_semantic_segmentation_loss: #  True,
            self.semantic_seg_conv = nn.Conv2d(src_channels[0], cfg.num_classes-1, kernel_size=1)  # (semantic_seg_conv): Conv2d(256, 80, kernel_size=(1, 1), stride=(1, 1))

        # For use in evaluation  评价函数中使用
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k,  #'nms_top_k': 200,
            conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)  
'''
    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS  剔除
    'nms_thresh': 0.5,
'''

    def save_weights(self, path): #存储权重
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)  #把模型对应的参数（权重和偏置weights，bias），存到对应的路径中   torch.save(MODEL.state_dict(), PATH)
    
    def load_weights(self, path): #载入储存的权重
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]
        
            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]

        self.load_state_dict(state_dict)

    def init_weights(self, backbone_path):
        """ Initialize weights for training. 如何设置权重参数的初始值关系到能否成功学习
        权重是用符合高斯分布的随机数进行初始化，偏置使用0进行初始化
        """
        # Initialize the backbone with the pretrained weights. 使用预训练的权重初始化基础网络
        self.backbone.init_backbone(backbone_path)  # 此函数init_backbone  初始化权重参数，在backbone中定义了

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')  #['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode', 'output_padding', 'in_channels', 'out_channels', 'kernel_size']
        
        # Quick lambda to test if one list contains the other，检测一个列表是否包含另一个列表
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier  这里我的理解就是，用  xavier初始值  初始化除了backbone之外的其他模块的权重
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Note that this might break with future pytorch updates, so let me know if it does              #  'Script' in type(module).__name__   这一项全部为False
            is_script_conv = 'Script' in type(module).__name__ \    
                and all_in(module.__dict__['_constants_set'], conv_constants) \
                and all_in(conv_constants, module.__dict__['_constants_set'])    #is_script_conv判断条件：module类型是'Script'，且conv_constants和module.__dict__['_constants_set']元素一样
            
            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv  #再加一条，module如果还是nn.Conv2d实例的话也算True
            
            if is_conv_layer and module not in self.backbone.backbone_modules:  #满足上述条件，并且不在backbone的部分，都用Xavier 均匀分布
                nn.init.xavier_uniform_(module.weight.data)  #如果前一层的节点数为n,则初始值使用标准差为1/root(n)的分布   
                
                if module.bias is not None: #True
                    if cfg.use_focal_loss and 'conf_layer' in name:  #cfg.use_focal_loss = False
                        if not cfg.use_sigmoid_focal_loss:   #'use_sigmoid_focal_loss': False,
                            # Initialize the last layer as in the focal loss paper.
                            # Because we use softmax and not sigmoid, I had to derive an alternate expression
                            # on a notecard. Define pi to be the probability of outputting a foreground detection.
                            # Then let z = sum(exp(x)) - exp(x_0). Finally let c be the number of foreground classes.
                            # Chugging through the math, this gives us
                            #   x_0 = log(z * (1 - pi) / pi)    where 0 is the background class
                            #   x_i = log(z / c)                for all i > 0
                            # For simplicity (and because we have a degree of freedom here), set z = 1. Then we have
                            #   x_0 =  log((1 - pi) / pi)       note: don't split up the log for numerical stability
                            #   x_i = -log(c)                   for all i > 0
                            module.bias.data[0]  = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0]  = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()   #初始偏置都给设置为0，深度学习入门上也是设置为0

    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:   #    'freeze_bn': False,
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): #判断module是不是nn.BatchNorm2d类
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] [batch_size, 3, 550, 550]""" 
        
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h #550
        cfg._tmp_img_w = img_w
        
        with timer.env('backbone'): #自定义的一个timer,That automatically manages a timer start and stop for you.
            outs = self.backbone(x) #输入数据先经过基础网络处理，输出的是tuple,
                                    # outs[0].size()=torch.Size([1,256,138,138]),outs[1]... 
                                    # #这里对应的输出是就是网络的C2-C5层，resnet101的输出

        if cfg.fpn is not None:
            with timer.env('fpn'):  #基础网络输出的数据，再经FPN层处理
                # Use backbone.selected_layers because we overwrote self.selected_layers 
                outs = [outs[i] for i in cfg.backbone.selected_layers] #[1,2,3] 对应的是C3,C4,C5
                outs = self.fpn(outs)  #torch.Size([1, 256, 69, 69]) torch.Size([1, 256, 35, 35]) torch.Size([1, 256, 18, 18]) torch.Size([1, 256, 9, 9]) torch.Size([1, 256, 5, 5]) 

        proto_out = None
        if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
            with timer.env('proto'):                                           #这一段是生成mask proto
                proto_x = x if self.proto_src is None else outs[self.proto_src] #这里是protonet的输入选择，为outs[0]
                
                if self.num_grids > 0: #self.num_grids=0
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out) #先self.proto_net()，后tanh激活，输出mask proto  
                #torch.Size([1, 32, 138, 138])

                if cfg.mask_proto_prototypes_as_features: #False
                    # Clone here because we don't want to permute this, though idk if contiguous makes this unnecessary
                    proto_downsampled = proto_out.clone()

                    if cfg.mask_proto_prototypes_as_features_no_grad: #False
                        proto_downsampled = proto_out.detach()
                
                # Move the features last so the multiplication is easy 将特征移到最后，这样乘法就容易了
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous() #torch.Size([1, 138, 138, 32])

                if cfg.mask_proto_bias:  #False
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)
        

        with timer.env('pred_heads'):
            pred_outs = { 'loc': [], 'conf': [], 'mask': [], 'priors': [] }

            if cfg.use_mask_scoring: #False
                pred_outs['score'] = []

            if cfg.use_instance_coeff: #False
                pred_outs['inst'] = []
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers): 
                #self.selected_layers = [0, 1, 2, 3, 4] prediction_layers就是五个PredictionModule的列表
                #这里的idx = [0, 1, 2, 3, 4] 
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features: #False
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                # A hack for the way dataparallel works       'share_prediction_module': True,
                if cfg.share_prediction_module and pred_layer is not self.prediction_layers[0]: #把1到4层的parent设置为self.prediction_layers[0]
                    pred_layer.parent = [self.prediction_layers[0]]

                p = pred_layer(pred_x) #预测模块的输出数据,输出的是一个字典，形式为{ 'loc': [], 'conf': [], 'mask': [], 'priors': [] }
                                       #  preds = { 'loc': bbox, 'conf': conf, 'mask': mask, 'priors': priors }
                for k, v in p.items():
                    pred_outs[k].append(v)  #把预测结果填到pred_outs中

        for k, v in pred_outs.items():
            pred_outs[k] = torch.cat(v, -2)  #上边得到的pred_outs中的值可能是这样的[tensor(),tensor(),...],这一步就是把这些在倒数第二个维度，摞成一个tensor,
                                            #torch.Size([1, 14283, 4]) torch.Size([1, 14283, 81])
        if proto_out is not None:  #把protonet的输出填入pred_outs中'proto'
            pred_outs['proto'] = proto_out

        if self.training:  #训练中的网络输出就是pred_layer的输出，
            # For the extra loss functions
            if cfg.use_class_existence_loss: #False
                pred_outs['classes'] = self.class_existence_fc(outs[-1].mean(dim=(2, 3)))

            if cfg.use_semantic_segmentation_loss: #False
                pred_outs['segm'] = self.semantic_seg_conv(outs[0])

            return pred_outs
'''
loc:  torch.Size([8, 19248, 4])
conf:  torch.Size([8, 19248, 11])
mask:  torch.Size([8, 19248, 32])
priors:  torch.Size([19248, 4])
proto:  torch.Size([8, 138, 138, 32])
'''
        else:   #eval中会把pred_layer的输出，经过边界框回归，NMS, 再输出
            if cfg.use_mask_scoring: #False
                pred_outs['score'] = torch.sigmoid(pred_outs['score'])

            if cfg.use_focal_loss:  #False
                if cfg.use_sigmoid_focal_loss:
                    # Note: even though conf[0] exists, this mode doesn't train it so don't use it
                    pred_outs['conf'] = torch.sigmoid(pred_outs['conf'])
                    if cfg.use_mask_scoring:
                        pred_outs['conf'] *= pred_outs['score']
                elif cfg.use_objectness_score:
                    # See focal_loss_sigmoid in multibox_loss.py for details
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    pred_outs['conf'][:, :, 1:] = objectness[:, :, None] * F.softmax(pred_outs['conf'][:, :, 1:], -1)
                    pred_outs['conf'][:, :, 0 ] = 1 - objectness
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)   
            else:

                if cfg.use_objectness_score: #False
                    objectness = torch.sigmoid(pred_outs['conf'][:, :, 0])
                    
                    pred_outs['conf'][:, :, 1:] = (objectness > 0.10)[..., None] \
                        * F.softmax(pred_outs['conf'][:, :, 1:], dim=-1)
                    
                else:
                    pred_outs['conf'] = F.softmax(pred_outs['conf'], -1) 
                    """
                    最后用softmax输出类别
                    """

            return self.detect(pred_outs)
#训练和评价时，yolact返回的不一样



# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()  #和控制台有关,这个可以先忽略不看

    # Use the first argument to set the config if you want  如果需要，请使用第一个参数设置配置
    import sys
    if len(sys.argv) > 1:  #tarin.py使用argparse从sys.argv中解析系统参数,python train.py --config=yolact_base_config, 系统的第二个参数即为网络配置参数,
        from data.config import set_cfg
        set_cfg(sys.argv[1])  #根据命令行命令 配置网络参数

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path) #载入权重

    # GPU
    net = net.cuda()
    cudnn.benchmark = True  #在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销，一般都会加。
    torch.set_default_tensor_type('torch.cuda.FloatTensor')   #设置默认GPUtensor类型

    x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))  #这里的x只是一个测试数据
    #输入网络的图像大小,创建1个3通道max_size*max_size的所有元素为0的tensor, N x C x H x W 的形式
    y = net(x) #y输出的是一个字典  { 'loc': [], 'conf': [], 'mask': [], 'priors': [], 'proto': [] }

    for p in net.prediction_layers: #预测头层，每层都打印一下这个层的输出特征图的大小
        print(p.last_conv_size)     #特征金字塔输出的大小

    print()
    for k, a in y.items():
        print(k + ': ', a.size(), torch.sum(a))   #loc:  conf:   mask:   priors:   proto:    segm:
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
