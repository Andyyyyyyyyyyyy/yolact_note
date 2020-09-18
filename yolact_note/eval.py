from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer  #这个timer用来计时
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.') #载入训练好的权重
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse') #如果再输入命令中设定了top_k，如15，那么每次做多显示15个目标
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.') #NMS快速版本，但是精度稍差
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.') #计算在不同类之间的NMS，我觉得应该设置为True
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes') #显示mask
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')  #显示BBOX
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])') #显示分类  以及分数
    parser.add_argument('--display_scores', default=True, type=str2bool, 
                        help='Whether or not to display scores in addition to classes') #显示分数
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')  #显示实例分割后的图像
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')  #跟--display一块的，display时图像是打乱来展示的
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.') #
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.') #从一个数据集中抽出多少张图片
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.') #把eval结果写入json文件中
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.') #把BBOX结果写入json文件
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.') #把mask结果写入json文件
    parser.add_argument('--config', default=None,
                        help='The config object to use.')  
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.') #默认eval的时候，会出现处理的进度条，这个就是不显示进度条，直接显示结果
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.') #要加上--display，先显示maskproto，然后是组合的maskproto,然后是分割结果图像
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.') 
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')  #如果不用边界框切断输出的mask的话，图像到处都是mask
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.') #
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.') #这里同时处理的视频帧大小，是不是就是相当于批大小
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.') #低置信概率目标不显示
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).') #设置的话就用这个数据集，而不用config中的验证集
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.') #不评估实例分割，只评估目标检测
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame') #处理视频，显示帧率
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:  #default=None
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]  #IOU门限：[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):   #prep_display(preds, img, h, w)
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)       #COCODetection输出的img是550*550的，这个函数是恢复原图像size,形状是h * w * c
        img_gpu = torch.Tensor(img_numpy).cuda()               #图像放到GPU上
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):   #这个t很重要！！！postprocess所输出的classes, scores, boxes, masks都已经按scores降序排序过了
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,  #这个是显示mask原型
                                        crop_masks        = args.crop,   
                                        score_threshold   = args.score_threshold)  # default=0,
        #classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)
        torch.cuda.synchronize()  #测试模型运行时间

    with timer.env('Copy'):  #
        


        if cfg.eval_mask_branch:  #'eval_mask_branch': True,
            # Masks are drawn on the GPU, so don't copy  #mask是在GPU上生成的，所以没有转化为numpy变量
            masks = t[3][:args.top_k] #default=5, 比如postprocess输出了87个masks，这里只取前五个，相应的，如果再输入命令中设定了top_k，如15，那么每次做多显示15个目标
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]  #取前top_k个元素，把类别，概率，边界框转化维numpy变量

    num_dets_to_consider = min(args.top_k, classes.shape[0])  #大多数情况下预测的类别是比较多的，这个要看权重的训练情况，
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:     #scores降序排序，要是前边的小于置信概率阈值，那么根本不用看后边的了
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):                                   #为筛选后的每个目标，快速选定一个颜色
        global color_cache   #color_cache = defaultdict(lambda: {})
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)   #选一个色号，COLORS的序号
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.   #放到GPU上的color被归一化
                color_cache[on_gpu][color_idx] = color
            return color
'''
# for making bounding boxes pretty 边界框的颜色
COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))
'''
    # First, draw the masks on the GPU where we can do it really fast   在GPU上画掩膜非常快
    # Beware: very fast but possibly unintelligible mask-drawing code ahead  这个mask绘制代码可能难以理解，但是非常快
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]  #形状加了一维
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])   #color本来是一维的，给改成了4维，形状变成了(1, 1, 1, 3)
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0) #本来是一个tensor列表，然后拼在一起变成了一个tensor
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha  #这里用repeat是因为，mask和colors形状不一致，所以用repeat来沿着指定的维度重复tensor，不易出错
                                                                      #masks.repeat(1, 1, 1, 3)  mask size变为(num_dets, h, w, 3),最后一列存储的是颜色信息
                                                                      #这个mask alpha 用于调节掩膜的透明度
                                                                      #masks_color  (num_dets, h, w, 3)
                                                                      #这里已经给mask安排上颜色了，其余地方是黑色的

        # This is 1 everywhere except for 1-mask_alpha where the mask is  mask所在的像素值为1-mask_alpha，其余变为1  (num_dets, h, w, 1)
        inv_alph_masks = masks * (-mask_alpha) + 1  
        '''
        这里说明一下，masks_color里边存储着一张图像上每个目标的带颜色的mask，每个目标对应一张添加颜色后的mask，没有颜色的部分像素值为0，有目标的部分，即mask部分，像素值用colormap表示
        而这个inv_alph_masks，是什么呢？他也有很多张，每张对应着一个目标的mask，mask部分呢是一个小数，用于调节透明度，其余的地方都是1，如果把他在0维乘起来，那么就把所有掩膜透明度放到了一张图上，其余部分都是1
        这样在去乘图像img_gpu的话，图像mask的对应部分颜色就会淡许多，确切讲是变黑，变暗，
        inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)这个的操作就是把第0维的后一项和前一项对应元素相乘，乘积作为后一项，这样乘过去之后，最后一项就等于前边所有项的乘积。
        在这里，inv_alph_masks[-1]就是一个包含所有mask的透明度调节器，mask部分等于1-mask_alpha（不相交的话），相交的话就更小了，其余部分都是1，
        这里我有点不明白为什么用masks_color[1:] * inv_alph_cumul，前者mask之外都是0，后者mask之外都是1，乘上之后masks_color也会变浅1-mask_alpha，
        masks_color0维的每一项对应一个mask,而inv_alph_masks0维序号越大，mask就+1，比如，masks_color[4]对应一个mask，但是inv_alph_masks对应四个mask，两者相乘之后还是只剩下一个mask,
        所以为什么用.cumprod(dim=0)，而不是masks_color[1:] * inv_alph_masks[1:]直接相乘？？？
        注意到[:(num_dets_to_consider-1)]   [1:]  用masks_color的第1项乘inv_alph_masks的第0项？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

        '''
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]   #形状(h, w, 3)
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)#.cumprod(dim=0) 在0维上，后面每行将前面行对应元素乘起来，就是把几个mask叠加在最后一维
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)   #输出的是带有mask色块的一张图

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand   #这里就是输出的带掩膜的图像  inv_alph_masks.prod(dim=0)是把num_dets个掩膜乘在一张图上形状变为 ( h, w, 1)
                                                                               #img_gpu形状为 ( h, w, 3)
    
    if args.display_fps:
            # Draw the box for the fps on the GPU
        font_face = cv2.FONT_HERSHEY_DUPLEX  #普通大小无衬线字体
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(fps_str, font_face, font_scale, font_thickness)[0]

        img_gpu[0:text_h+8, 0:text_w+8] *= 0.6 # 1 - Box alpha


    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()  #恢复到255

    if args.display_fps:
        # Draw the text on the CPU
        text_pt = (4, text_h + 2)
        text_color = [255, 255, 255]

        cv2.putText(img_numpy, fps_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)  #显示文字设置
    
    if num_dets_to_consider == 0:
        return img_numpy

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):  #倒序
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1) #画一个BBOX 图像，左上角坐标，右下角坐标，颜色，后边的1是无填充

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0] #这个就是我写上去的这个文字，计算他占的宽和高

                text_pt = (x1, y1 - 3)    #这个是文字的原点，在哪标注文字
                text_color = [255, 255, 255]   

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)  #画一个文字框，填充颜色
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA) #把文字放上去
            
    
    return img_numpy  #返回加上类别名，边界框等的图像

def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k].cpu().numpy() for x in t]
    
    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()

def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:

    def __init__(self):
        self.bbox_data = [] #这里是为输出json文件服务的  --output_coco_json 会输出bbox_detections.json  mask_detections.json
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]  #改为左上角坐标+宽，高的形式

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests 按照COCO的建议，四舍五入到最接近的10，以避免文件过大,有什么效果？？？？
        bbox = [round(float(x)*10)/10 for x in bbox]    #结果类似这种[34.0, 89.0, 50.0, 50.0]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),  #
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, args.bbox_det_file),  ##args.bbox_det_file 这是路径   把BBOX结果写入json文件
            (self.mask_data, args.mask_det_file)   ##把Mask结果写入json文件
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
        

        

def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()

#prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections) ap_data计算每个IOU门限下每个类的AP， preds=net(batch),
def prep_metrics(ap_data, dets, img, gt, gt_masks, h, w, num_crowd, image_id, detections:Detections=None): #Detections=None  不把输出文件写入json
    """ Returns a list of APs for this image, with each element being for a class 这个函数主要用来根据预测结果和gt来填充ap_data """
    if not args.output_coco_json:
        with timer.env('Prepare gt'):
            gt_boxes = torch.Tensor(gt[:, :4])
            gt_boxes[:, [0, 2]] *= w                #这里是准备BBOX 真值，乘上宽高，变为原图大小
            gt_boxes[:, [1, 3]] *= h
            gt_classes = list(gt[:, 4].astype(int))   #gt的最后一列是对应的类别，用数字表示
            gt_masks = torch.Tensor(gt_masks).view(-1, h*w)  #把mask gt 拉伸成一维的，长度：图片长*宽

            if num_crowd > 0:
                split = lambda x: (x[-num_crowd:], x[:-num_crowd])
                crowd_boxes  , gt_boxes   = split(gt_boxes)
                crowd_masks  , gt_masks   = split(gt_masks)
                crowd_classes, gt_classes = split(gt_classes)

    with timer.env('Postprocess'):
        classes, scores, boxes, masks = postprocess(dets, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)  #dets=preds  preds=net(batch)
        #输出的都是GPU tensor
        if classes.size(0) == 0:
            return

        classes = list(classes.cpu().numpy().astype(int)) #把GPU类型数据转换为numpy数据
        scores = list(scores.cpu().numpy().astype(float))
        masks = masks.view(-1, h*w).cuda()
        boxes = boxes.cuda()


    if args.output_coco_json:
        with timer.env('JSON Output'):
            boxes = boxes.cpu().numpy()
            masks = masks.view(-1, h, w).cpu().numpy()
            for i in range(masks.shape[0]):         #一张图片中的所有检测目标，挨个填到json文件
                # Make sure that the bounding box actually makes sense and a mask was produced
                if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:    #边界框 长乘宽  大于0
                    detections.add_bbox(image_id, classes[i], boxes[i,:],   scores[i])
                    detections.add_mask(image_id, classes[i], masks[i,:,:], scores[i])
            return
    
    with timer.env('Eval Setup'):
        num_pred = len(classes)   #预测的类别数
        num_gt   = len(gt_classes)  #真值的类别数

        mask_iou_cache = _mask_iou(masks, gt_masks)  #mask都是被拉成一维，计算maskIOU
        bbox_iou_cache = _bbox_iou(boxes.float(), gt_boxes.float())

        if num_crowd > 0:
            crowd_mask_iou_cache = _mask_iou(masks, crowd_masks, iscrowd=True)  
            crowd_bbox_iou_cache = _bbox_iou(boxes.float(), crowd_boxes.float(), iscrowd=True)
        else:
            crowd_mask_iou_cache = None
            crowd_bbox_iou_cache = None

        iou_types = [
            ('box',  lambda i,j: bbox_iou_cache[i, j].item(), lambda i,j: crowd_bbox_iou_cache[i,j].item()),
            ('mask', lambda i,j: mask_iou_cache[i, j].item(), lambda i,j: crowd_mask_iou_cache[i,j].item())
        ]


    timer.start('Main loop')
    for _class in set(classes + gt_classes):   # 集合（set）是一个无序的不重复元素序列，可以使用大括号 { } 或者 set() 函数创建集合，
        ap_per_iou = []
        num_gt_for_class = sum([1 for x in gt_classes if x == _class])  #没看懂？？？
        
        for iouIdx in range(len(iou_thresholds)):  ##IOU门限：[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
            iou_threshold = iou_thresholds[iouIdx]

            for iou_type, iou_func, crowd_func in iou_types:  #'box' 'mask'
                gt_used = [False] * len(gt_classes)
                
                ap_obj = ap_data[iou_type][iouIdx][_class]
                ap_obj.add_gt_positives(num_gt_for_class)

                for i in range(num_pred):  ##预测的类别数
                    if classes[i] != _class:
                        continue
                    
                    max_iou_found = iou_threshold
                    max_match_idx = -1
                    for j in range(num_gt):
                        if gt_used[j] or gt_classes[j] != _class:
                            continue
                            
                        iou = iou_func(i, j)

                        if iou > max_iou_found:
                            max_iou_found = iou
                            max_match_idx = j
                    
                    if max_match_idx >= 0:
                        gt_used[max_match_idx] = True
                        ap_obj.push(scores[i], True)
                    else:
                        # If the detection matches a crowd, we can just ignore it
                        matched_crowd = False

                        if num_crowd > 0:
                            for j in range(len(crowd_classes)):
                                if crowd_classes[j] != _class:
                                    continue
                                
                                iou = crowd_func(i, j)

                                if iou > iou_threshold:
                                    matched_crowd = True
                                    break

                        # All this crowd code so that we can make sure that our eval code gives the
                        # same result as COCOEval. There aren't even that many crowd annotations to
                        # begin with, but accuracy is of the utmost importance.
                        if not matched_crowd:
                            ap_obj.push(scores[i], False)
    timer.stop('Main loop')


class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class. 存储计算AP的对象信息
    Note: I type annotated this because why not.

    AP衡量的是对一个类检测好坏，mAP就是对多个类的检测好坏。就是简单粗暴的把所有类的AP值取平均就好了。比如有两类，类A的AP值是0.5，类B的AP值是0.3，那么mAP=（0.5+0.3）/2=0.4
    怎么计算AP？
    1、计算出bounding-box与ground-Truth的IoU的值，利用阈值把预选框分为TP&FP 
    2、按置信度从大到小把检测结果排列，计算累计精度和召回率，如：第一个检测结果是TP，一共15个groundtruth，那么P就是1，recall就是1/15；第二个是FP，那么P就是1/(1+1)，recall还是1/15.
       第三个检测结果是TP，P=2/3 R=2/15， 第四个FP，P=2/4 R=2/15 ，这样这几个图像的检测结果一直累加完，然后每一个检测结果都对应着一个precision 和 recall，这样就可以画出P-R图
    3. 每个recall对应着可能不止一个precision,取最大的那个；然后，要是有P-R曲线凹下去的情况，就把那部分“填平”。这样处理之后开始计算AP，这里把recall从0到1分成101个数，[0,0.01,0.02,...,0.99,1]
       但是实际的recall值可能没有这么多，可能对不上，那么就把没有recall的地方用右边的recall对应的precision 顶上
    这就是AP的计算实现
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0 #ground truth个数

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))  #这里是每个预测输出的目标的置信度，+  是TP,还是FP 
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives   #添加每张图片的ground truth目标个数

    def is_empty(self) -> bool:  #这个符号 ->  python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型，
        return len(self.data_points) == 0 and self.num_gt_positives == 0  #返回bool值  判断是不是空的


    #计算出bounding-box与ground-Truth的IoU的值，利用阈值把预选框分为TP&FP
    def get_ap(self) -> float:     #计算AP值
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0  

        # Sort descending by score # 把检测结果按置信度降序排序
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0  #True Positive  检测出来的真值
        num_false = 0  #False Positive  检测出来的假值

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:  #数据结构应该是这样的[(0.95, True),(0.83,False),(0.75,False),()...]  
            # datum[1] is whether the detection a true or false positive  #  datum[1]是bool值
            if datum[1]: num_true += 1    #计算TP和PN的个数
            else: num_false += 1
            
            precision = num_true / (num_true + num_false) #每一个数据计算一下精度Precision 和 Recall
            recall    = num_true / self.num_gt_positives  #召回率为TP/所有真值个数，  precision-recall曲线下的面积是AP  

            precisions.append(precision)  
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        # 就是较小的recall值对应的precision只能大于等于较大的recall对应的
        for i in range(len(precisions)-1, 0, -1):  #这个意思就是这个曲线不能有凹下去的地方，得是个凸曲线，
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars. recall从0到1均分成101份，用黎曼求和计算AP
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)]) # 0到1，步长0.01  array([0.  , 0.01, 0.02, 0.03, 0.04,  ...  , 0.99, 1.  ])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) 
        # = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')  #在数组a中插入数组v（并不执行插入操作），返回一个下标列表，这个列表指明了v中对应元素应该插入在a中那个位置上
        for bar_idx, precision_idx in enumerate(indices):         #np.searchsorted([1,2,3], 2.1, side='left')->[1,2,2.1,3] 序号 2 
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]  #填充没有的recall对应的precision

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)  #AP是在0和1之间的所有召回值的平均精度。   

def badhash(x):
    """
    Just a quick and dirty hash function for doing a deterministic shuffle based on image_id.

    Source:
    https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    """
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF   # ^ 按位异或 ,先右移16位，再异或，然后乘，最后和1与，输出一个十进制数
    x = (((x >> 16) ^ x) * 0x045d9f3b) & 0xFFFFFFFF
    x =  ((x >> 16) ^ x) & 0xFFFFFFFF
    return x

def evalimage(net:Yolact, path:str, save_path:str=None):  #处理单张图片 输入，输出
    frame = torch.from_numpy(cv2.imread(path)).cuda().float()  #numpy——>tensor,再到cuda
    batch = FastBaseTransform()(frame.unsqueeze(0)) #unsqueeze在0维增加一个维度,squeeze减少一个维度
    preds = net(batch)

    img_numpy = prep_display(preds, frame, None, None, undo_transform=False)
    
    if save_path is None:
        img_numpy = img_numpy[:, :, (2, 1, 0)]

    if save_path is None:
        plt.imshow(img_numpy)
        plt.title(path)
        plt.show()
    else:
        cv2.imwrite(save_path, img_numpy)

def evalimages(net:Yolact, input_folder:str, output_folder:str): #处理一堆图片
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    print()
    for p in Path(input_folder).glob('*'): 
        path = str(p)
        name = os.path.basename(path)
        name = '.'.join(name.split('.')[:-1]) + '.png'
        out_path = os.path.join(output_folder, name)

        evalimage(net, path, out_path)
        print(path + ' -> ' + out_path)
    print('Done.')

from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])

def evalvideo(net:Yolact, path:str, out_path:str=None):  #处理视频
    # If the path is a digit, parse it as a webcam index
    is_webcam = path.isdigit()
    
    # If the input image size is constant, this make things faster (hence why we can use it in a video setting).
    cudnn.benchmark = True
    
    if is_webcam:
        vid = cv2.VideoCapture(int(path))
    else:
        vid = cv2.VideoCapture(path)
    
    if not vid.isOpened():
        print('Could not open video "%s"' % path)
        exit(-1)

    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames   = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if is_webcam:
        num_frames = float('inf')

    net = CustomDataParallel(net).cuda()
    transform = torch.nn.DataParallel(FastBaseTransform()).cuda()
    frame_times = MovingAverage(100)
    fps = 0
    frame_time_target = 1 / target_fps
    running = True
    fps_str = ''
    vid_done = False
    frames_displayed = 0

    if out_path is not None:
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))

    def cleanup_and_exit():
        print()
        pool.terminate()
        vid.release()
        if out_path is not None:
            out.release()
        cv2.destroyAllWindows()
        exit()

    def get_next_frame(vid):
        frames = []
        for idx in range(args.video_multiframe):
            frame = vid.read()[1]
            if frame is None:
                return frames
            frames.append(frame)
        return frames

    def transform_frame(frames):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float() for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def eval_network(inp):
        with torch.no_grad():
            frames, imgs = inp
            num_extra = 0
            while imgs.size(0) < args.video_multiframe:
                imgs = torch.cat([imgs, imgs[0].unsqueeze(0)], dim=0)
                num_extra += 1
            out = net(imgs)
            if num_extra > 0:
                out = out[:-num_extra]
            return frames, out

    def prep_frame(inp, fps_str):
        with torch.no_grad():
            frame, preds = inp
            return prep_display(preds, frame, None, None, undo_transform=False, class_color=True, fps_str=fps_str)

    frame_buffer = Queue()
    video_fps = 0

    # All this timing code to make sure that 
    def play_video():
        nonlocal frame_buffer, running, video_fps, is_webcam, num_frames, frames_displayed, vid_done

        video_frame_times = MovingAverage(100)
        frame_time_stabilizer = frame_time_target
        last_time = None
        stabilizer_step = 0.0005
        progress_bar = ProgressBar(30, num_frames)

        while running:
            frame_time_start = time.time()

            if not frame_buffer.empty():
                next_time = time.time()
                if last_time is not None:
                    video_frame_times.add(next_time - last_time)
                    video_fps = 1 / video_frame_times.get_avg()
                if out_path is None:
                    cv2.imshow(path, frame_buffer.get())
                else:
                    out.write(frame_buffer.get())
                frames_displayed += 1
                last_time = next_time

                if out_path is not None:
                    if video_frame_times.get_avg() == 0:
                        fps = 0
                    else:
                        fps = 1 / video_frame_times.get_avg()
                    progress = frames_displayed / num_frames * 100
                    progress_bar.set_val(frames_displayed)

                    print('\rProcessing Frames  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                        % (repr(progress_bar), frames_displayed, num_frames, progress, fps), end='')

            # Press Escape to close
            if cv2.waitKey(1) == 27 or not (frames_displayed < num_frames):
                running = False

            if not vid_done:
                buffer_size = frame_buffer.qsize()
                if buffer_size < args.video_multiframe:
                    frame_time_stabilizer += stabilizer_step
                elif buffer_size > args.video_multiframe:
                    frame_time_stabilizer -= stabilizer_step
                    if frame_time_stabilizer < 0:
                        frame_time_stabilizer = 0

                new_target = frame_time_stabilizer if is_webcam else max(frame_time_stabilizer, frame_time_target)
            else:
                new_target = frame_time_target

            next_frame_target = max(2 * new_target - video_frame_times.get_avg(), 0)
            target_time = frame_time_start + next_frame_target - 0.001 # Let's just subtract a millisecond to be safe
            
            if out_path is None or args.emulate_playback:
                # This gives more accurate timing than if sleeping the whole amount at once
                while time.time() < target_time:
                    time.sleep(0.001)
            else:
                # Let's not starve the main thread, now
                time.sleep(0.001)


    extract_frame = lambda x, i: (x[0][i] if x[1][i] is None else x[0][i].to(x[1][i]['box'].device), [x[1][i]])

    # Prime the network on the first frame because I do some thread unsafe things otherwise
    print('Initializing model... ', end='')
    first_batch = eval_network(transform_frame(get_next_frame(vid)))
    print('Done.')

    # For each frame the sequence of functions it needs to go through to be processed (in reversed order)
    sequence = [prep_frame, eval_network, transform_frame]
    pool = ThreadPool(processes=len(sequence) + args.video_multiframe + 2)
    pool.apply_async(play_video)
    active_frames = [{'value': extract_frame(first_batch, i), 'idx': 0} for i in range(len(first_batch[0]))]

    print()
    if out_path is None: print('Press Escape to close.')
    try:
        while vid.isOpened() and running:
            # Hard limit on frames in buffer so we don't run out of memory >.>
            while frame_buffer.qsize() > 100:
                time.sleep(0.001)

            start_time = time.time()

            # Start loading the next frames from the disk
            if not vid_done:
                next_frames = pool.apply_async(get_next_frame, args=(vid,))
            else:
                next_frames = None
            
            if not (vid_done and len(active_frames) == 0):
                # For each frame in our active processing queue, dispatch a job
                # for that frame using the current function in the sequence
                for frame in active_frames:
                    _args =  [frame['value']]
                    if frame['idx'] == 0:
                        _args.append(fps_str)
                    frame['value'] = pool.apply_async(sequence[frame['idx']], args=_args)
                
                # For each frame whose job was the last in the sequence (i.e. for all final outputs)
                for frame in active_frames:
                    if frame['idx'] == 0:
                        frame_buffer.put(frame['value'].get())

                # Remove the finished frames from the processing queue
                active_frames = [x for x in active_frames if x['idx'] > 0]

                # Finish evaluating every frame in the processing queue and advanced their position in the sequence
                for frame in list(reversed(active_frames)):
                    frame['value'] = frame['value'].get()
                    frame['idx'] -= 1

                    if frame['idx'] == 0:
                        # Split this up into individual threads for prep_frame since it doesn't support batch size
                        active_frames += [{'value': extract_frame(frame['value'], i), 'idx': 0} for i in range(1, len(frame['value'][0]))]
                        frame['value'] = extract_frame(frame['value'], 0)
                
                # Finish loading in the next frames and add them to the processing queue
                if next_frames is not None:
                    frames = next_frames.get()
                    if len(frames) == 0:
                        vid_done = True
                    else:
                        active_frames.append({'value': frames, 'idx': len(sequence)-1})

                # Compute FPS
                frame_times.add(time.time() - start_time)
                fps = args.video_multiframe / frame_times.get_avg()
            else:
                fps = 0
            
            fps_str = 'Processing FPS: %.2f | Video Playback FPS: %.2f | Frames in Buffer: %d' % (fps, video_fps, frame_buffer.qsize())
            if not args.display_fps:
                print('\r' + fps_str + '    ', end='')

    except KeyboardInterrupt:
        print('\nStopping...')
    
    cleanup_and_exit()

def evaluate(net:Yolact, dataset, train_mode=False): #train_mode是否在训练中
    net.detect.use_fast_nms = args.fast_nms   #default=True, 'Whether to use a faster, but not entirely correct version of NMS.'
    net.detect.use_cross_class_nms = args.cross_class_nms  #default=False
    cfg.mask_proto_debug = args.mask_proto_debug   #mask_proto_debug=False

    if args.image is not None: #处理单张图片
        if ':' in args.image:
            inp, out = args.image.split(':')  #--image=input_image.png:output_image.png针对有输出的图像
            evalimage(net, inp, out)
        else:
            evalimage(net, args.image)    #--image=input_image.png
        return
    elif args.images is not None:  #处理图片集
        inp, out = args.images.split(':')  #--images=path/to/input/folder:path/to/output/folder
        evalimages(net, inp, out)
        return
    elif args.video is not None:
        if ':' in args.video:
            inp, out = args.video.split(':')   #--video=input_video.mp4:output_video.mp4
            evalvideo(net, inp, out)
        else:
            evalvideo(net, args.video)  #--video=my_video.mp4   --video=0
        return

    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))   #max_images   default=-1，这里要是没有设置最大张数，就测试整个数据集的数据
    progress_bar = ProgressBar(30, dataset_size)  #这应该是 处理过程中进度条的长度设置

    print()

    if not args.display and not args.benchmark:   #display=False， benchmark=False  
        # For each class and iou, stores tuples (score, isPositive)
        # Index ap_data[type][iouIdx][classIdx]
        ap_data = {
            'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],  #计算每个IOU门限下每个类的AP
            'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
        }  #
        detections = Detections() #把输出写入json文件
    else:
        timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))  #测试集或验证集  元素序列
    
    if args.shuffle:  #shuffle=False
        random.shuffle(dataset_indices)
    elif not args.no_sort:  #default=False
        # Do a deterministic shuffle based on the image ids 根据图像id做确定性打乱？？？？，打乱后的顺序是确定的
        #
        # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
        # the order of insertion. That means on python 3.6, the images come in the order they are in
        # in the annotations file. For some reason, the first images in the annotations file are
        # the hardest. To combat this, I use a hard-coded hash function based on the image ids
        # to shuffle the indices we use. That way, no matter what python version or how pycocotools
        # handles the data, we get the same result every time.
        hashed = [badhash(x) for x in dataset.ids]   #datasetid反映的就是标注文件中的图像ID，hash后id变成了一堆很大的十进制数，对应着原来的id，然后把这些十进制数升序排序，把对应的id也这样挪到相应的位置
        dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]  #这里就是把image给打乱，不过这个打乱的顺序始终是固定的

    try:
        # Main eval loop  这才是主循环
        for it, image_idx in enumerate(dataset_indices):  
            timer.reset()
'''
#每次处理验证集中的一张图片！！！
'''
            with timer.env('Load Data'):
                img, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)  #这个dataset是COCODetection载入的dataset，包含各种信息
                                                                                   #dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                                                                                        # transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
                #img是tensor，img_size=(3,550,550)；     gt是边界框，形状(n,5),前四列是BBOX坐标，最后一列为数字，代表这个BBOX中目标的类别； 
                #   gt_mask形状为N*375*1242,我认为n就是目标个数，然后对应着图像的size，
                #这里的gt_mask四个通道对应着四个标注的目标，每一个通道里有一个标注的目标，这个目标的值为1，其余都为0

                # Test flag, do not upvote
                if cfg.mask_proto_debug:  #default=False,
                    with open('scripts/info.txt', 'w') as f:
                        f.write(str(dataset.ids[image_idx]))
                    np.save('scripts/gt.npy', gt_masks)


                batch = Variable(img.unsqueeze(0)) #验证集中全部的图像，在0维给img加上一个维度，(1, 3, 550, 550)
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                preds = net(batch)                #送到网络中前向传播，preds输出
'''
        yolact网络的输出
        输出：    形状：
        classes   n      n为目标个数
        scores    n
        boxes     n*4    
        masks     n*32   这里的masks只是一个系数而已，再与proto矩阵相乘后形状变为  138*138*32
        proto     138*138*32
'''
            # Perform the meat of the operation here depending on our mode.  根据选择的模式，执行相应代码，最重要的部分
            if args.display:
                img_numpy = prep_display(preds, img, h, w)
            elif args.benchmark:
                prep_benchmark(preds, h, w)
            else:
                prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.ids[image_idx], detections)
            
            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:  #意思就是不算第一个的处理时间，因为第一个的时间比较慢
                frame_times.add(timer.total_time())
            
            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)                 
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
                #cv2.imshow("cv2_img",cv2.cvtColor(img_numpy,cv2.COLOR_RGB2BGR))  太麻烦！
                #cv2.imshow("cv2_img",img_numpy[:, :, (2, 1, 0)])

                #cv2.waitKey(0)
            elif not args.no_bar:
                if it > 1: fps = 1 / frame_times.get_avg()  #所有的
                else: fps = 0
                progress = (it+1) / dataset_size * 100  #当前进度
                progress_bar.set_val(it+1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                    % (repr(progress_bar), it+1, dataset_size, progress, fps), end='')



        if not args.display and not args.benchmark:
            print()
            if args.output_coco_json:
                print('Dumping detections...')
                if args.output_web_json:
                    detections.dump_web()
                else:
                    detections.dump()
            else:
                if not train_mode:
                    print('Saving data...')
                    with open(args.ap_data_file, 'wb') as f:
                        pickle.dump(ap_data, f)

                return calc_map(ap_data)    #计算所有阈值，所有类下的mAP  #训练的时候计算mAP
        elif args.benchmark:
            print()
            print()
            print('Stats for the last frame:')
            timer.print_stats()
            avg_seconds = frame_times.get_avg()
            print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')


def calc_map(ap_data):
    print('Calculating mAP...')
    aps = [{'box': [], 'mask': []} for _ in iou_thresholds]  #IOU门限：[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for _class in range(len(cfg.dataset.class_names)):  #class
        for iou_idx in range(len(iou_thresholds)):   #10
            for iou_type in ('box', 'mask'):
                ap_obj = ap_data[iou_type][iou_idx][_class]

                if not ap_obj.is_empty():
                    aps[iou_idx][iou_type].append(ap_obj.get_ap())

    all_maps = {'box': OrderedDict(), 'mask': OrderedDict()}

    # Looking back at it, this code is really hard to read :/
    for iou_type in ('box', 'mask'):
        all_maps[iou_type]['all'] = 0 # Make this first in the ordereddict
        for i, threshold in enumerate(iou_thresholds):
            mAP = sum(aps[i][iou_type]) / len(aps[i][iou_type]) * 100 if len(aps[i][iou_type]) > 0 else 0
            all_maps[iou_type][int(threshold*100)] = mAP
        all_maps[iou_type]['all'] = (sum(all_maps[iou_type].values()) / (len(all_maps[iou_type].values())-1))
    
    print_maps(all_maps)
    
    # Put in a prettier format so we can serialize it to json during training
    all_maps = {k: {j: round(u, 2) for j, u in v.items()} for k, v in all_maps.items()}
    return all_maps

def print_maps(all_maps):
    # Warning: hacky 
    make_row = lambda vals: (' %5s |' * len(vals)) % tuple(vals)
    make_sep = lambda n:  ('-------+' * n)

    print()
    print(make_row([''] + [('.%d ' % x if isinstance(x, int) else x + ' ') for x in all_maps['box'].keys()]))
    print(make_sep(len(all_maps['box']) + 1))
    for iou_type in ('box', 'mask'):
        print(make_row([iou_type] + ['%.2f' % x if x < 100 else '%.1f' % x for x in all_maps[iou_type].values()]))
    print(make_sep(len(all_maps['box']) + 1))
    print()



if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':  #trained_model就是权重，根据这个调用不同的载入函数
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'   #yolact_base
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:   #不评估实例分割，只评估目标检测
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.resume and not args.display:
            with open(args.ap_data_file, 'rb') as f:
                ap_data = pickle.load(f)
            calc_map(ap_data)
            exit()

        if args.image is None and args.video is None and args.images is None:  #如果没有设定图像，视频或者图像文件夹，就拿验证集的图像来测试
            dataset = COCODetection(cfg.dataset.valid_images, cfg.dataset.valid_info,
                                    transform=BaseTransform(), has_gt=cfg.dataset.has_gt)
            prep_coco_cats()
        else:
            dataset = None        

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)


