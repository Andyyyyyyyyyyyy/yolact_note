# coding=utf-8

from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")  #字符串型 转 布尔型, 字符串做小写处理后,若是 "yes", "true", "t", "1" 中的一个,则返回True,否则False


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training') #可选参数, 批大小,默认8
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.') #恢复训练
parser.add_argument('--start_iter', default=-1, type=int,
                    help='Resume training at this iter. If this is -1, the iteration will be'\
                         'determined from the file name.') #恢复训练之后的初始迭代次数,若为-1,则从权重文件名中获取
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading') # dataloading 数据加载  中使用的工作线程数
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model') #用CUDA来训练模型
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')#初始学习率,不要管,会自动从config.py中获取
parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')# Momentum ,SGD权重更新的算法,自动从config.py中获取
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')#权值衰减,对大的权重进行惩罚,来抑制过拟合,自动从config.py中获取
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.') #gamma,去乘lr，自动从config.py中获取
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models.')#权重 保存的文件夹,这里的checkpoint就应该是每迭代多少次后我存一下训练好的权重文件
parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.') # 训练 记录
parser.add_argument('--config', default=None,
                    help='The config object to use.') #
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.') #保存间隔,隔多少次迭代（处理一个batch为一次迭代）,保存一次
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')  #用于验证的图片数量
parser.add_argument('--validation_epoch', default=2, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.') # 每2次迭代输出一次验证信息
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.') #dest关键字:指定储存变量，  只保存最新的权重存档
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')#在这声明数据集,就不管config文件中的声明的数据集了
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.') #不记录迭代信息,一次iteration迭代就是处理一个批的数据
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')#在日志中包含GPU信息。Nvidia-smi往往很慢，所以请谨慎设置
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')#捕获键盘中断时不要保存权重
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).') #batch,一次处理的图片数量,一批里边安排给各个GPU处理的图片数量，加一块要等于批大小
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')#禁用 根据批大小自动调节学习率和迭代数量

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)  #set_cfg函数是config.py里的, 训练命令:python train.py --config=yolact_base_config
                          #这一步的目的就是   载入整个网络的配置

if args.dataset is not None:
    set_dataset(args.dataset)  #这个函数也在config.py里

if args.autoscale and args.batch_size != 8:   #批大小只要不是8就执行这句,说明  args.autoscale  始终是True,见84行
    factor = args.batch_size / 8
    print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))  

    cfg.lr *= factor    #cfg参数来自config.py,    用factor自动缩放网络的配置参数,学习率,最大迭代次数,学习率步长
    cfg.max_iter //= factor  #整除,根据factor缩放最大迭代次数
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]

# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))  #如果在args中没有设置这些参数,那么就直接从cfg中获取
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0: #返回gpu数量
    print('No GPUs detected. Exiting...')
    exit(-1)  #退出当前运行的程序，并将参数-1返回给主调进程

if args.batch_size // torch.cuda.device_count() < 6:  #若设定的批大小除以GPU数量<6,则禁用 批正规化BN 功能, #网上说法:bn在小batch size条件下很不稳定，不如不用，
    print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S']

if torch.cuda.is_available():  #cuda是否可用；
    if args.cuda:  #True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')  #torch.set_default_tensor_type设置默认的tensor的数据类型的函数
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")    #先警告一下
        torch.set_default_tensor_type('torch.FloatTensor')    #继续不用cuda的话,就用cpu的tensor类型
else:
    torch.set_default_tensor_type('torch.FloatTensor')    #上边if中查看可用GPU=0的话就退出了,这里说没有cuda可用就用cpu的tensor类型

class NetLoss(nn.Module):  
    """
        此函数用于   前向传递+计算损失  datum
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.

    net = CustomDataParallel(NetLoss(net, criterion)) 
    losses = net(datum)
    NetLoss中已经设置好了net和criterion，

    net(datum)，我觉得是先把datum经过CustomDataParallel.scatter来生成NetLoss.forward所需的images, targets, masks, num_crowds
    然后把数据送到NetLoss.forward跑出预
    测结果preds，然后在送到MultiBoxLoss这个里边计算一下损失
    最后通过CustomDataParallel.gather将计算的损失输出
    这样就说的通了，不知道对不对

    """
    
    def __init__(self, net:Yolact, criterion:MultiBoxLoss):  #设置网络和评价指标,//from yolact import Yolact //from layers.modules import MultiBoxLoss
        super().__init__()

        self.net = net
        self.criterion = criterion
    
    def forward(self, images, targets, masks, num_crowds):  #前向计算
        preds = self.net(images)  #images形状 batch_size 3 550 550
        return self.criterion(preds, targets, masks, num_crowds)  #这里返回的损失是一个字典数据
"""
        # Loss Key:
        #  - B: Box Localization Loss
        #  - C: Class Confidence Loss
        #  - M: Mask Loss
        #  - P: Prototype Loss
        #  - D: Coefficient Diversity Loss
        #  - E: Class Existence Loss
        #  - S: Semantic Segmentation Loss
        return losses
"""

class CustomDataParallel(nn.DataParallel): #pytorch多GPU训练,这个应该是和设置每个GPU处理的图像数量有关
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):  #分散
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids] #['cuda:0','cuda:1']大电脑上就两个 GPU 0,1
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)   #准备数据，把datum中的images, targets, masks, num_crowds送给对应的GPU，组成一个4维的列表
"""
这里的inputs[0]是datum，那完整的inputs是什么???
举例说明：splits=([tensor([]) ,2 ],[31 ,42 ],[51 ,62 ],[71 ,82 ])
splits里的数据都是根据分配到GPU上的数据堆叠成的，如split_images[0].shape()=[10,3,550,550]
split_images        [None ,None ]   #行代表images，targets等，列对应的是第几个GPU上的数据，里边元素的大小和batch_size有关，比如batch_size是2，那么splits[0][0].shape()=(2, 3, 550, 550)
split_targets       [None ,None ]
split_masks         [None ,None ]
split_numcrowds     [None ,None ]
"""
        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices) #返回两个值,一个是二维列表,行对应每个GPU,列是分配的参数;第二个值不用说了
            #splits=([1 ,2 ],[31 ,42 ],[51 ,62 ],[71 ,82 ])，则return [[1, 31, 51, 71], [2, 42, 62, 82]]

    def gather(self, outputs, output_device): #聚集
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])
        
        return out

def train():
    if not os.path.exists(args.save_folder):  #训练时如果没有保存权重的文件夹,就创建ddd
        os.mkdir(args.save_folder)
    #COCODetection在data/coco.py中定义
    dataset = COCODetection(image_path=cfg.dataset.train_images,  #训练集的路径，在config中设置
                            info_file=cfg.dataset.train_info,     #注释信息
                            transform=SSDAugmentation(MEANS))  #SSDAugmentation应该是数据扩充的函数,论文里也说了用SSD的数据增广方法
    
    if args.validation_epoch > 0:   #validation_epoch=2 每2次迭代，输出一次验证信息
        setup_eval()
        val_dataset = COCODetection(image_path=cfg.dataset.valid_images, #载入验证集
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))

    # Parallel wraps the underlying module, but when saving and loading we don't want that  并行包装基础模块，但是在保存和加载时我们不希望这样
    yolact_net = Yolact() #yolact函数
    net = yolact_net
    net.train()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),   #from utils.logger import Log 
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)  #我理解的意思,如果不从终断点开始记录args.resume is None,那么就执行覆盖函数

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check    
    #如果不给resume指定权重文件,那么如果令--resume=interrupt,则会从weights/下的interrupt权重开始恢复训练,最新的interrupt权重会覆盖旧的
    #这样之后就给args.resume指定了权重文件,然后下边yolact_net.load_weights再载入该文件
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)    #from utils.functions import MovingAverage, SavePath
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:   # default=None 
        print('Resuming training, loading {}...'.format(args.resume)) #.format():把传统的%替换为{}来实现格式化输出
        yolact_net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration   #start_iter=-1,从权重文件中获取迭代次数
    else:
        print('Initializing weights...')
        yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path) #如果不指定resume权重就从初始权重开始训练

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)    #import torch.optim as optim   权值更新(优化)方法
    criterion = MultiBoxLoss(num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)     #损失函数计算,IOU=0.5,ohem_negpos_ratio=3

    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]    #argparse默认输入变量是string,所以这里把args.batch_alloc当成字符串来处理
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size)) #给每个GPU分配的批图像数量加一块不等于批大小,则报错
            exit(-1)

    net = CustomDataParallel(NetLoss(net, criterion))
    if args.cuda:  #True,用CUDA来训练模型
        net = net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda())
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size    #epoch_size为训练集图片总数除以批大小,一个epoch有多少个batch
    num_epochs = math.ceil(cfg.max_iter / epoch_size) #按照所设定的最大迭代次数,需要迭代这么多个epoch
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,  #加载数据的时候使用几个子进程
                                  shuffle=True, collate_fn=detection_collate,  
                                  pin_memory=True)   #DataLoader为pytorch的数据加载API，在此之前需要首先将数据包装为Dataset类，然后传入DataLoader中，这里是载入数据集和人工标签
    
    #一个epoch大小等于整个训练集的图像数量, epoch,iteration是lambda函数的参数
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)#保存的名字和路径,如weight/yolact_base_2666_80000.pth
    time_avg = MovingAverage()

    global loss_types # Forms the print order   loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S']
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }
    # Loss Key:
    #  - B: Box Localization Loss
    #  - C: Class Confidence Loss
    #  - M: Mask Loss
    #  - P: Prototype Loss
    #  - D: Coefficient Diversity Loss
    #  - E: Class Existence Loss
    #  - S: Semantic Segmentation Loss
    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training #键盘ctrl+c中断训练,保存已训练的权重
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter    #这句主要是考虑到会恢复之前中断的地方开始训练,此时start_iter>0,continue直接执行下一个for循环,直到epoch到对应的迭代次数,才执行下边的语句
            if (epoch+1)*epoch_size < iteration:   #iteration是可以设置的
                continue
            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter #不明白这一句???????????????????????????????,这里直接跳出循环了
                if iteration == (epoch+1)*epoch_size: 
                    break

                # Stop at the configured number of iterations even if mid-epoch 
                #到达设定的最大迭代次数,停止!
                if iteration == cfg.max_iter:
                    break

                #对应config.py中,达到设定的iteration之后就对config执行相应的改变
                # Change a config setting if we've reached the specified iteration  
                changed = False  #标志位  
                for change in cfg.delayed_settings:  #这个delayed_setting在config中并没有设置  
                    if iteration >= change[0]:  
                        changed = True  
                        cfg.replace(change[1])  

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration] #delayed_settings中大于本次iteration的会保留,已经执行过的就删了

                # Warm up by linearly interpolating the learning rate from some smaller value 
                #'lr_warmup_until': 500,  在500个迭代之后,初始学习率增长到基础学习率
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                #config.py中'lr_steps': (280000, 600000, 700000, 750000),也要根据factor缩放 'gamma': 0.1, 在这几个迭代后,要调整学习率大小,降低学习率
                #lr' = lr * gamma ^ step_index
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                
                # Zero the grad to get ready to compute gradients  #把梯度清零，好计算这个迭代的梯度
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss) #前向传递+计算损失,下边都是计算损失,然后打印损失
                losses = net(datum)
                
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop  反向传播
                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item(): #判断是不是收敛了
                    optimizer.step()  #更新权重
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time  #第一个last_time是开始时间
                last_time = cur_time

                # Exclude graph setup from the timing information    #计算每个循环的平均时间
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:   #每10个iteration显示一次,#max_iter根据factor缩放后的
                    eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]   #这个预计时间可能是根据最大迭代次数和当前迭代次数的差值,乘上每个迭代大概用多久,算出来的
                    
                    total = sum([loss_avgs[k].get_avg() for k in losses])  #total应该指的是总共的损失
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                    
                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f')
                            % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(losses[k].item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)  #这些都写到log文件中了

                    log.log_gpu_stats = args.log_gpu  #记录GPU信息比较慢，默认不记录
                
                iteration += 1  #把一次迭代的损失都计算完了,才+1,准备进行下次迭代

                if iteration % args.save_interval == 0 and iteration != args.start_iter: #这里就是每隔1W次迭代，保存当前权重一次
                    if args.keep_latest: #默认设置keep_latest=False，保存权重时，只保留最新的权重
                        latestyolact_net = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))#保存权重文件

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            
            # This is done per epoch #跟深度学习入门书中例程一样，每个iter计算一次loss,这里两个epoch计算针对整个的精度，在计算精度时要把所有数据都代入计算，所以比较慢，所以两个epoch计算一次
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:   #两个epoch计算一次mAP
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)  #用验证集来计算mAP
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder 覆盖掉之前的interrupt权重文件
            SavePath.remove_interrupt(args.save_folder)
            
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))    #保存当前中断时的权重文件,并退出
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))#所有的执行完了再保存一遍???  这个yolact_net就包括backbone，fpn,proto_net，prediction_layers和semantic_seg_conv这些部分，保存对应的参数


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr #当前学习率
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):  #prepare_data(inputs[0], devices, allocation=args.batch_alloc) 如( *,['cuda:0','cuda:1'],[10,6])
    with torch.no_grad():  #在autograd里不跟踪这次 weight - grad的运算  #with torch.no_grad()或者@torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
        if devices is None: 
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:  #没有设置分配给各个GPU的图片数量的话，就在这平均分配一下
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)  #这里用到了列表的重复,如[32//4]*(4-1) >>>[8,8,8]
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum  #num_crowds=[0]

        cur_idx = 0
        #2. 这里先处理了一下images, targets, masks, num_crowds,应该是把图片安排到处理它的GPU,和这个有关x.requires_grad = False
        for device, alloc in zip(devices, allocation):  #zip的作用,把列表对应元素组合起来,用括号括起来，如:[('cuda:0',10),('cuda:1',6)]
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))  #images[0]  = gradinator(images[0].to('cuda:0')) batch内的每个image都对应自己的编号
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))  #这里就是把这些cpu变量变为CUDA变量
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1   #cur_idx最后加完了为batch_size-1

        if cfg.preserve_aspect_ratio:  #False  3. 缩放时,保持图像纵横比, config.py中这个preserve_aspect_ratio设置为False,就是缩放时直接把图像拉伸成550*550的
            # Choose a random size from the batch  这里我的理解：比如batchsize是16，那么images里边也有16个图，然后在0-15中随机选一个序号,输出对应图片的size 2020.04.30@peng
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)   #这里估计就是缩放图片了
        
        cur_idx = 0
        #4. 创建二维列表来容纳准备好的数据
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]  #创建一个元素为None的len(allocation)列4行的二维列表,前边的4个变量对应列表的4行,如allocation为[10,6],则取其长度2,取for循环,得到4*2的列表
"""
split_images        [None ,None ]  两个GPU对应两个None
split_targets       [None ,None ]
split_masks         [None ,None ]
split_numcrowds     [None ,None ]
"""
        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)#在0维，把分配到同一个GPU的图像堆叠，作为split_images的一个元素
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds  

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):  #计算验证集损失
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in data_loader:
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):   #计算验证集mAP
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    train()
