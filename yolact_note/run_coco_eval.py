# coding=utf-8

"""
这个程序的主要作用是,首先用eval.py提取COCO数据集训练的权重中的各种目标的框和掩膜的json文件
bbox_detections.json和mask_detections.json, 然后用这个程序将COCO数据集标注的json基准文件,与训练后经权重提取的json文件做对比,
然后输出一个检测结果

Runs the coco-supplied cocoeval script to evaluate detections
outputted by using the output_coco_json flag in eval.py.
"""


import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser(description='COCO Detections Evaluator')
parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str)
parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str)
parser.add_argument('--gt_ann_file',   default='data/coco/annotations/instances_val2017.json', type=str)
parser.add_argument('--eval_type',     default='both', choices=['bbox', 'mask', 'both'], type=str) #eval_type的值从['bbox', 'mask', 'both']三个里边选,超出范围报错
args = parser.parse_args()



if __name__ == '__main__':

	eval_bbox = (args.eval_type in ('bbox', 'both'))  #判断args.eval_type中有没有'bbox'或者 'both',有,True;否则False.若python run_coco_eval.py --eval_type mask 则False
	eval_mask = (args.eval_type in ('mask', 'both'))

	print('Loading annotations...')
	gt_annotations = COCO(args.gt_ann_file)
	if eval_bbox:
		bbox_dets = gt_annotations.loadRes(args.bbox_det_file)
	if eval_mask:
		mask_dets = gt_annotations.loadRes(args.mask_det_file)

	if eval_bbox:
		print('\nEvaluating BBoxes:')
		bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()
	
	if eval_mask:
		print('\nEvaluating Masks:')
		bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
		bbox_eval.evaluate()
		bbox_eval.accumulate()
		bbox_eval.summarize()



