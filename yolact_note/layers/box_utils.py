# -*- coding: utf-8 -*-
import torch
from utils import timer

from data import cfg

@torch.jit.script
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


@torch.jit.script
def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h

@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)   #目标数量
    A = box_a.size(1)   #
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd:bool=False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]   [num_objects, num_priors]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]  #计算面积就是用乘法
    area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)

def elemwise_box_iou(box_a, box_b):
    """ Does the same as above but instead of pairwise, elementwise along the inner dimension. """
    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, 0] * inter[:, 1]

    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    union = area_a + area_b - inter
    union = torch.clamp(union, min=0.1)

    # Return value is [n] for inputs [n, 4]
    return torch.clamp(inter / union, max=1)

def mask_iou(masks_a, masks_b, iscrowd=False):
    """
    Computes the pariwise mask IoU between two sets of masks of size [a, h, w] and [b, h, w].
    The output is of size [a, b].

    Wait I thought this was "box_utils", why am I putting this in here?
    """

    masks_a = masks_a.view(masks_a.size(0), -1)  #形状为预测到的mask数 * （w*h）
    masks_b = masks_b.view(masks_b.size(0), -1)  #形状为gt的mask数 * （w*h）

    intersection = masks_a @ masks_b.t()   #masks_b.t()转置   pytorch  @是用来对tensor进行矩阵相乘的
    area_a = masks_a.sum(dim=1).unsqueeze(1)  #每一行相加，再把和append到一维tensor
    area_b = masks_b.sum(dim=1).unsqueeze(0)

    return intersection / (area_a + area_b - intersection) if not iscrowd else intersection / area_a

def elemwise_mask_iou(masks_a, masks_b):
    """ Does the same as above but instead of pairwise, elementwise along the outer dimension. """
    masks_a = masks_a.view(-1, masks_a.size(-1))
    masks_b = masks_b.view(-1, masks_b.size(-1))

    intersection = (masks_a * masks_b).sum(dim=0)
    area_a = masks_a.sum(dim=0)
    area_b = masks_b.sum(dim=0)

    # Return value is [n] for inputs [h, w, n]
    return torch.clamp(intersection / torch.clamp(area_a + area_b - intersection, min=0.1), max=1)



def change(gt, priors):
    """
    Compute the d_change metric proposed in Box2Pix:
    https://lmb.informatik.uni-freiburg.de/Publications/2018/UB18/paper-box2pix.pdf
    
    Input should be in point form (xmin, ymin, xmax, ymax).

    Output is of shape [num_gt, num_priors]
    Note this returns -change so it can be a drop in replacement for 
    """
    num_priors = priors.size(0)
    num_gt     = gt.size(0)

    gt_w = (gt[:, 2] - gt[:, 0])[:, None].expand(num_gt, num_priors)
    gt_h = (gt[:, 3] - gt[:, 1])[:, None].expand(num_gt, num_priors)

    gt_mat =     gt[:, None, :].expand(num_gt, num_priors, 4)
    pr_mat = priors[None, :, :].expand(num_gt, num_priors, 4)

    diff = gt_mat - pr_mat
    diff[:, :, 0] /= gt_w
    diff[:, :, 2] /= gt_w
    diff[:, :, 1] /= gt_h
    diff[:, :, 3] /= gt_h

    return -torch.sqrt( (diff ** 2).sum(dim=2) )




def match(pos_thresh, neg_thresh, truths, priors, labels, crowd_boxes, loc_t, conf_t, idx_t, idx, loc_data):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        pos_thresh: (float) IoU > pos_thresh ==> positive.
        neg_thresh: (float) IoU < neg_thresh ==> negative.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].                #边界框真值
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].           #预选框
        labels: (tensor) All the class labels for the image, Shape: [num_obj].            #类别真值
        crowd_boxes: (tensor) All the crowd box annotations or None if there are none.    #None
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.                 #这个要填上边界框回归后的边界框
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. Note: -1 means neutral.
        idx_t: (tensor) Tensor to be filled w/ the index of the matched gt box for each prior.   #
        idx: (int) current batch index. 
        loc_data: (tensor) The predicted bbox regression coordinates for this batch.      #边界框回归时的系数，预测头输出的loc
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    #use_prediction_matching: False
    decoded_priors = decode(loc_data, priors, cfg.use_yolo_regressors) if cfg.use_prediction_matching else point_form(priors)  #point_form：把xc,yc,w,h转化为 xmin, ymin, xmax, ymax这种形式的bbox坐标 
    #在训练中，用anchor产生的预选框和真值相比较，IOU大于等于某个阈值时，认为是positive

    # Size [num_objects, num_priors]    use_change_matching: False   计算每个边界框真值和anchor边界框之间的IOU矩阵
    overlaps = jaccard(truths, decoded_priors) if not cfg.use_change_matching else change(truths, decoded_priors)

    # Size [num_priors] best ground truth for each prior  #把这张图像中，几个真值目标，对应的最大IOU的priors预选框的IOU值，和序号输出来，每一列的最大值和列序号，一共19248个
    best_truth_overlap, best_truth_idx = overlaps.max(0)   #  find the max overlap gt with each anchor


    # We want to ensure that each gt gets used at least once so that we don't
    # waste any training data. In order to do that, find the max overlap anchor
    # with each gt, and force that anchor to use that gt.
    for _ in range(overlaps.size(0)):   #一张图片中的目标个数
        # Find j, the gt with the highest overlap with a prior
        # In effect, this will loop through overlaps.size(0) in a "smart" order,
        # always choosing the highest overlap first.
        best_prior_overlap, best_prior_idx = overlaps.max(1)   #与上边正好相反，这里是在anchor预选框中找出与每一个gtbbox重合最大的,best_prior_idx 0 ~ 19248
        # 例：一个overlaps为9*19248的矩阵
        #best_prior_overlap： tensor([0.4362, 0.5132, 0.4654, 0.5247, 0.3768, 0.4878, 0.4212, 0.5470, 0.4408])   
        # tensor([16434, 19113, 18537, 16526, 16592, 19094, 16320, 16341, 18532]) 

        j = best_prior_overlap.max(0)[1]  #最大IOU的序号   0 ~ (num_objects-1)   j是best_prior_overlap中的最大值的序号
        #tensor(7) 

        # Find i, the highest overlap anchor with this gt   再找出最大IOU序号对应的anchor预选框，i是j对应的best_prior_idx的序号
        i = best_prior_idx[j]
        #tensor(16341)

        # Set all other overlaps with i to be -1 so that no other gt uses it
        overlaps[:, i] = -1
        # Set all other overlaps with j to be -1 so that this loop never uses j again   #把最大IOU那行和列，都给设置成-1,在接下来的循环中，best_prior_overlap中对应的项变为-1
        overlaps[j, :] = -1
        # tensor([0.4362, 0.5132, 0.4654, 0.5247, 0.3768, 0.4878, 0.4212, -1, 0.4408])   
        # tensor([16434, 19113, 18537, 16526, 16592, 19094, 16320, 0, 18532]) 
        # 循环完了之后，overlaps都变为-1

        # Overwrite i's score to be 2 so it doesn't get thresholded ever  # 然后把best_truth_overlap中的值设置为2，编号这些的值16434, 19113, 18537, 16526, 16592, 19094, 16320, 16341, 18532
        best_truth_overlap[i] = 2
        # Set the gt to be used for i to be j, overwriting whatever was there  #然后序号给改了，对应到j
        best_truth_idx[i] = j 

    matches = truths[best_truth_idx]            # Shape: [num_priors,4]   truth为tensor([[],[],...]) 这里按下标从truths中取19248个坐标list
    conf = labels[best_truth_idx] + 1           # Shape: [num_priors]   #因为labels中的类别是从0开始的，而下边如果小于设定阈值，就给赋0，所以此处+1，避免出现错误

    conf[best_truth_overlap < pos_thresh] = -1  # label as neutral
    conf[best_truth_overlap < neg_thresh] =  0  # label as background

    # Deal with crowd annotations for COCO
    if crowd_boxes is not None and cfg.crowd_iou_threshold < 1:
        # Size [num_priors, num_crowds]
        crowd_overlaps = jaccard(decoded_priors, crowd_boxes, iscrowd=True)
        # Size [num_priors]
        best_crowd_overlap, best_crowd_idx = crowd_overlaps.max(1)
        # Set non-positives with crowd iou of over the threshold to be neutral.
        conf[(conf <= 0) & (best_crowd_overlap > cfg.crowd_iou_threshold)] = -1

    loc = encode(matches, priors, cfg.use_yolo_regressors)
    loc_t[idx]  = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf   # [num_priors] top class label for each prior
    idx_t[idx]  = best_truth_idx # [num_priors] indices for lookup

@torch.jit.script
def encode(matched, priors, use_yolo_regressors:bool=False):
    """
    Encode bboxes matched with each prior into the format
    produced by the network. See decode for more details on
    this format. Note that encode(decode(x, p), p) = x.
    相当于计算预测头输出中的loc，即我理解的边界框回归系数
    Args:
        - matched: A tensor of bboxes in point form with shape [num_priors, 4]
        - priors:  The tensor of all priors with shape [num_priors, 4]
    Return: A tensor with encoded relative coordinates in the format
            outputted by the network (see decode). Size: [num_priors, 4]
    """

    if use_yolo_regressors:
        # Exactly the reverse of what we did in decode
        # In fact encode(decode(x, p), p) should be x
        boxes = center_size(matched)   #Convert prior_boxes to (cx, cy, w, h)

        loc = torch.cat((
            boxes[:, :2] - priors[:, :2],
            torch.log(boxes[:, 2:] / priors[:, 2:])
        ), 1)
    else:
        variances = [0.1, 0.2]

        # dist b/t match center and prior's center
        g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
        # encode variance
        g_cxcy /= (variances[0] * priors[:, 2:])
        # match wh / prior wh
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        # return target for smooth_l1_loss
        loc = torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
        
    return loc

@torch.jit.script
def decode(loc, priors, use_yolo_regressors:bool=False):
    """
    decode(loc_data[batch_idx], prior_data)
    loc:  torch.Size([1, 19248, 4])
    priors:  torch.Size([19248, 4])
    
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

    Decode predicted bbox coordinates using the same scheme   解码bbox坐标
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x    如果use_yolo_regressors=True，前半部分sigmoid(pred_x) - .5) / conv_w已经在yolact.py中执行了
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)
    
    Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are inputed as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the convouts.
    
    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).
    
    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]  网络预测的bbox
        - priors: The priorbox coords with size [num_priors, 4]         anchor机制下的预选框
    
    Returns: A tensor of decoded relative coordinates in point form 
             form with size [num_priors, 4]
    """

    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = torch.cat((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * torch.exp(loc[:, 2:])  #跟上边的式子类似
        ), 1)

        boxes = point_form(boxes)
        '''
@torch.jit.script
def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax   可以理解，就是由中心点的坐标减去、加上长宽的一半，得到(xmin, ymin, xmax, ymax)这种形式的bbox坐标
        '''
    else:
        variances = [0.1, 0.2]  方差
        
        boxes = torch.cat((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]               #这一块跟上边的point_form差不多，都是得到(xmin, ymin, xmax, ymax)
    
    return boxes   



def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1)) + x_max


@torch.jit.script
def sanitize_coordinates(_x1, _x2, img_size:int, padding:int=0, cast:bool=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1-padding, min=0)
    x2 = torch.clamp(x2+padding, max=img_size)

    return x1, x2


@torch.jit.script
def crop(masks, boxes, padding:int=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)
    
    masks_left  = rows >= x1.view(1, 1, -1)
    masks_right = rows <  x2.view(1, 1, -1)
    masks_up    = cols >= y1.view(1, 1, -1)
    masks_down  = cols <  y2.view(1, 1, -1)
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask.float()


def index2d(src, idx):
    """
    Indexes a tensor by a 2d index.

    In effect, this does
        out[i, j] = src[i, idx[i, j]]
    
    Both src and idx should have the same size.
    """

    offs = torch.arange(idx.size(0), device=idx.device)[:, None].expand_as(idx)
    idx  = idx + offs * idx.size(1)

    return src.view(-1)[idx.view(-1)].view(idx.size())
