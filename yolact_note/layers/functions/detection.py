import torch
import torch.nn.functional as F
from ..box_utils import decode, jaccard, index2d
from utils import timer

from data import cfg, mask_type

import numpy as np


class Detect(object):
    """
    # For use in evaluation  评价函数中使用
    self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=cfg.nms_top_k, conf_thresh=cfg.nms_conf_thresh, nms_thresh=cfg.nms_thresh)  

    # Examples with confidence less than this are not considered by NMS
    'nms_conf_thresh': 0.05,
    # Boxes with IoU overlap greater than this threshold will be culled during NMS  剔除
    'nms_thresh': 0.5,

    At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh   #0.5
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh    #0.05
        
        self.use_cross_class_nms = False   #我觉得这个有必要是True， 这两个是在eval.py中设置的，use_fast_nms = True
        self.use_fast_nms = False

    def __call__(self, predictions):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        { 'loc': [], 'conf': [], 'mask': [], 'priors': [], 'proto': [] }

例：batchsize = 1, 19248个anchor机制下的预选框
loc:  torch.Size([1, 19248, 4])
conf:  torch.Size([1, 19248, 11])
mask:  torch.Size([1, 19248, 32])
priors:  torch.Size([19248, 4])
proto:  torch.Size([1, 138, 138, 32])

 decode:   
关于边界框回归，我的理解，就是把anchor机制得到的预选框，通过平移中心坐标和缩放长宽，得到一个bbox，这个bbox很接近groundtruth bbox，
这里priors就不用说了，是anchor机制得到的预选框； 而loc，说是叫bbox，其实不是，更像是一个平移和缩放系数，因为他有负的，而priors都是大于0的，这样就可以把priors缩放和平移
loc:  
tensor([[[-0.3929,  1.1011, -2.3080,  0.0654],
         [-1.2304, -0.3642, -4.5283, -2.1763],
         [ 0.1369,  0.8213, -1.7665,  0.4082],
         ...,
         [ 0.2331,  0.3178, -1.3481, -0.4864],
         [ 0.4684,  0.2520, -1.9428, -0.6018],
         [ 0.2811,  0.4669, -1.1389, -0.3000]]]) 
priors:
 tensor([[0.0072, 0.0072, 0.0436, 0.0436],
        [0.0072, 0.0072, 0.0309, 0.0309],
        [0.0072, 0.0072, 0.0617, 0.0617],
        ...,
        [0.9000, 0.9000, 0.6982, 0.6982],
        [0.9000, 0.9000, 0.4937, 0.4937],
        [0.9000, 0.9000, 0.9874, 0.9874]])


        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()  #conf:  torch.Size([1, 11, 19248])

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)  #torch.Size([1, 19248, 4])  边界框回归
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]
                
                out.append(result)
        
        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]  #去掉背景class的conf
        conf_scores, _ = torch.max(cur_scores, dim=0)   #shape 19824  这里是用的列（代表检测结果）的最大值，这个最大值要是都小于conf_thresh，那么这个检测结果就可以丢掉

        keep = (conf_scores > self.conf_thresh)  #一维的bool型变量
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]   #过滤掉99%的网络输出，由19248降到几十

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
    
        if scores.size(1) == 0:
            return None
        
        if self.use_fast_nms:
            if self.use_cross_class_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            else:
                boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh)

            if self.use_cross_class_nms:
                print('Warning: Cross Class Traditional NMS is not implemented.')

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}


    def cc_fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)

        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]

        boxes_idx = boxes[idx]

        # Compute the pairwise IoU between the boxes
        iou = jaccard(boxes_idx, boxes_idx)
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        iou.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the IoU matrix along the columns, each column will represent the maximum IoU
        # between this element and every element with a higher score than this element.
        iou_max, _ = torch.max(iou, dim=0)

        # Now just filter out the ones greater than the threshold, i.e., only keep boxes that
        # don't have a higher scoring box that would supress it in normal NMS.
        idx_out = idx[iou_max <= iou_threshold]
        
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True) #按置信度大小降序排序   shape: 10*dete

        idx = idx[:, :top_k].contiguous()     #取前200个  在经过一次过滤之后，基本上都是在200以内
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()    #10*dete

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)    #idx.view(-1)平铺为一维，boxes由num_dets * 4变为num_classes * num_dets * 4 ,这里是根据idx来索引，相当于每一类的det对应的bbox
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)   #masks由num_dets * 32 变为 num_classes * num_dets * 32 

        iou = jaccard(boxes, boxes)  #计算IOU？？？，先计算每个类下的boxes之间的IOU，形成一个对角矩阵
        iou.triu_(diagonal=1)        #取对角阵的右上部分
        iou_max, _ = iou.max(dim=1)  #右上部分，每一列的最大值

        # Now just filter out the ones higher than the threshold 类间IOU最大值<=iou阈值的，保留
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)  #    'max_num_detections': 100,
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.size(0)    #类别数目

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):  #每个类别挨个NMS
            cls_scores = scores[_cls, :]    #一维 
                                            # 这里又用了conf_thresh，detect里的是为了快速过滤calss_num*19248个结果，这里是对过滤后的每个检测结果，再过滤;
                                             # 即，过滤每一个类别下的检测结果，
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)  #int型tensor，从0到每个类对应的检测数-1

            cls_scores = cls_scores[conf_mask]      
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:     #这里就有很多类里边是空的了
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()   #把class scores加到box后面
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()     #先变成numpy变量，然后NMS，再变回GPU tensor，list

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)    #类别的list就是这么来的
            scr_lst.append(cls_scores[keep])

        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]      #  'max_num_detections': 100,
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores
