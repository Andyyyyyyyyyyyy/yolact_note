# -*- coding: UTF-8 -*- 

import json
import os
import sys
from collections import defaultdict

#抽取部分或者全部注释数据,先存到out中,然后在写入out_name

usage_text = """
This script creates a coco annotation file by mixing one or more existing annotation files.
																								创建一个新的json注释文件,从原来的注释文件中抽取选定的注释
Usage: python data/scripts/mix_sets.py output_name [set1 range1 [set2 range2 [...]]]

To use, specify the output annotation name and any number of set + range pairs, where the sets
are in the form instances_<set_name>.json and ranges are python-evalable ranges. The resulting
json will be spit out as instances_<output_name>.json in the same folder as the input sets.

For instance,
    python data/scripts/mix_sets.py trainval35k train2014 : val2014 :-5000  					":-5000"从第一个元素到最后一个元素的前5000个元素 之间的元素
																								y = x[:] 通过分片操作将列表x的元素全部拷贝给y
This will create an instance_trainval35k.json file with all images and corresponding annotations
from train2014 and the first 35000 images from val2014.

You can also specify only one set:
    python data/scripts/mix_sets.py minival5k val2014 -5000:

This will take the last 5k images from val2014 and put it in instances_minival5k.json.
"""

annotations_path = 'data/coco/annotations/instances_%s.json'
fields_to_combine = ('images', 'annotations')
fields_to_steal   = ('info', 'categories', 'licenses')

if __name__ == '__main__':
	if len(sys.argv) < 4 or len(sys.argv) % 2 != 0: 				#系统参数必须要大于4,否则报错,python data/scripts/mix_sets.py minival5k val2014 -5000: 包括data/scripts/mix_sets.py一共4个参数
		print(usage_text)   										#打印"使用方法",并且退出
		exit()

	out_name = sys.argv[1]  										#输出名称为第二个系统参数
	sets = sys.argv[2:]     										#从第三个元素到最后一个元素,e.g.,['val2014', '-5000:', 'train2014', ':']
	sets = [(sets[2*i], sets[2*i+1]) for i in range(len(sets)//2)] 	#   //为整数除法,e.g.,[('val2014', '-5000:'), ('train2014', ':')]
	
	out = {x: [] for x in fields_to_combine}  						#out = {'images': [], 'annotations': []}

	for idx, (set_name, range_str) in enumerate(sets):  			#枚举,list(enumerate(sets))为[(0, ('val2014', '-5000:')), (1, ('train2014', ':'))],列出数据序号和数据
		print('Loading set %s...' % set_name)
		with open(annotations_path % set_name, 'r') as f:  			#如,打开data/coco/annotations/instances_val2014.json
			set_json = json.load(f)    								#json.load()用于从json文件中读取数据。  json.loads()用于将str类型的数据转成dict。

		# "Steal" some fields that don't need to be combined from the first set  “偷”一些第一组不需要合并的字段
		if idx == 0:
			for field in fields_to_steal:
				out[field] = set_json[field]  						#out{'images': [], 'annotations': [], 'info': {},'categories': [],'licenses': []}

		print('Building image index...')
		image_idx = {x['id']: x for x in set_json['images']} 		#set_json['images']=[{'license': 3, ...},{'license': 1, ...},...] // x={'license': 3, ...}... //x['id'] = 9...
																	#全部数据!!!image_idx = 9:{'license': 3, ...}
		print('Collecting annotations...')
		anns_idx = defaultdict(lambda: [])							#使用defaultdict任何未定义的key都会默认返回一个根据method_factory参数不同的默认值, 而相同情况下dict()会返回KeyError
																	#defaultdict(lambda: [])创建了一个字典anns_idx，默认值是[]
		for ann in set_json['annotations']:
			anns_idx[ann['image_id']].append(ann)					#全部数据!!!同一张图片中，不同目标的标注，176470: [{'segmentation': [[415.48, 71.24, 430.08, 73.87, 434.63, 87.28, 427.45, 99.97, 414.52, 107.63, 404.47, 106.67, 398.72, 91.83, 402.31, 80.57, 409.02, 74.59, 415.0, 71.72]], 'area': 934.1901499999992, 'iscrowd': 0, 'image_id': 176470, 'bbox': [398.72, 71.24, 35.91, 36.39], 'category_id': 85, 'id': 2231901}, {'segmentation': [[475.84, 78.89, 484.15, 87.45, 488.17, 97.53, 486.92, 111.63, 482.63, 114.65, 476.34, 109.36, 474.07, 96.02, 473.57, 83.42]], 'area': 351.39900000000046, 'iscrowd': 0, 'image_id': 176470, 'bbox': [473.57, 78.89, 14.6, 35.76], 'category_id': 85, 'id': 2231908}]
		export_ids = list(image_idx.keys())
		export_ids.sort()											#全部数据!!!图片序号升序排序
		export_ids = eval('export_ids[%s]' % range_str, {}, {'export_ids': export_ids})   #只输出指定ID的数据,eval() 函数用来执行一个字符串表达式，并返回表达式的值。  高级
		
		print('Adding %d images...' % len(export_ids))
		for _id in export_ids:										#给out{'images': [], 'annotations': []}填上值,只把对应ID的数据填充上去
			out['images'].append(image_idx[_id])
			out['annotations'] += anns_idx[_id]
		print('Done.\n')
		
	print('Saving result...')
	with open(annotations_path % (out_name), 'w') as out_file:		#将out中的数据都存到out_name里边,这个文件是在此处建立的,直接新建一个文件,用json.dump(数据,输出文件)命令
		json.dump(out, out_file)
