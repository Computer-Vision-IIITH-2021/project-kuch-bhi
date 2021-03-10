import os
from xmlparse import parseXML
from selective_search import find_region_proposals
import cv2
import numpy as np
from tqdm import tqdm

def IoU(bbox1, bbox2):
	x_left = max(bbox1[0],bbox2[0])
	y_top = max(bbox1[1],bbox2[1])
	x_right = max(bbox1[2],bbox2[2])
	y_bottom = max(bbox1[3],bbox2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
	area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])

	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	union_area = area1 + area2 - intersection_area
	return intersection_area/union_area

def main():
	images_path = "../../../VOC2012/JPEGImages"
	annots_path = "../../../VOC2012/Annotations"
	out_path = "../../../training_data"
	if not os.path.exists(out_path):
		os.mkdir(out_path)

	out_filename = lambda class_count_, class_index_: os.path.join(out_path, f'{class_index_}_{class_count_}.jpg')

	NUM_IMAGES = 10
	LIMIT_CLASS_IMAGES = 30
	image_size = (224,224)

	class_list = ['background','person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','diningtable','pottedplant','sofa','tvmonitor']
	NUM_CLASSES = len(class_list)
	class_index_mapping = {}
	for classname in class_list:
		class_index_mapping[classname] = class_list.index(classname)

	count = 0
	class_count = np.zeros((NUM_CLASSES), dtype='int')
	total_class_count = class_count.copy()
	for filename in tqdm(os.listdir(images_path)[:NUM_IMAGES]):
		xml_filepath = os.path.join(annots_path, filename.split('.')[0] + '.xml') 
		image_filepath = os.path.join(images_path, filename)
		class_count *= 0

		classnames, boxes = parseXML(xml_filepath)
		image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

		for bbox in find_region_proposals(image):
			x1,y1,x2,y2 = bbox
			max_iou = 0
			for classname, box in zip(classnames, boxes):
				class_index = class_index_mapping[classname]
				iou = IoU(bbox,box)
				max_iou = max(max_iou,iou)
				if iou > 0.7 and class_count[class_index]<LIMIT_CLASS_IMAGES:
					cv2.imwrite(out_filename(total_class_count[class_index],class_index), cv2.resize(image[y1:y2,x1:x2], image_size))
					class_count[class_index] += 1
					total_class_count[class_index] += 1
			if max_iou < 0.3 and class_count[0]<LIMIT_CLASS_IMAGES:
				cv2.imwrite(out_filename(total_class_count[0],0), cv2.resize(image[y1:y2,x1:x2], image_size))
				class_count[0] += 1
				total_class_count[0] += 1

if __name__=="__main__":
	main()