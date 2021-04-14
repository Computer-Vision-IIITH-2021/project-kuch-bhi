import os
from xmlparse import parseXML
from selective_search import find_region_proposals
import json
import cv2
import numpy as np
from tqdm import tqdm

roi_dict = {}
image_size = (224,224)

def IoU(bbox1, bbox2, return_coords=False):
	x_left = max(bbox1[0],bbox2[0])
	y_top = max(bbox1[1],bbox2[1])
	x_right = min(bbox1[2],bbox2[2])
	y_bottom = min(bbox1[3],bbox2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0, (0,0,0,0)

	area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
	area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])

	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	union_area = area1 + area2 - intersection_area
	if return_coords:
		return intersection_area/union_area, (x_left, y_top, x_right, y_bottom)
	return intersection_area/union_area

labelify = lambda class_index_, coords_: f'{class_index_}_{"_".join(map(str,coords_))}'
normalize_roi_1000 = lambda coords_, size: (np.round(coords_/np.array([size[1],size[0],size[1],size[0]]), 3)*1000).astype('int')
roi_to_str = lambda coords_: "_".join(map(str,coords_))

def box_write(image_filename, roi, class_index, bbox, size):
	if image_filename not in roi_dict:
		roi_dict[image_filename] = []
	label = labelify(class_index, bbox)
	roi_str = roi_to_str(normalize_roi_1000(roi,size))
	roi_dict[image_filename].append(roi_str+"_"+label)

def save_rois(path):
	to_remove = []
	for filename in roi_dict:
		if len(roi_dict[filename])<8:
			to_remove.append(filename)
	for filename in to_remove:
		roi_dict.pop(filename)
	with open(path, 'w') as f:
		json.dump(roi_dict, f)

def main():
	images_path = "../../../VOC2012/JPEGImages"
	annots_path = "../../../VOC2012/Annotations"
	out_path = "../../../training_rois.json"

	NUM_IMAGES = 10
	LIMIT_CLASS_IMAGES = 5

	class_list = ['background','person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','diningtable','pottedplant','sofa','tvmonitor']
	NUM_CLASSES = len(class_list)
	class_index_mapping = {}
	for classname in class_list:
		class_index_mapping[classname] = class_list.index(classname)

	count = 0
	class_count = np.zeros((NUM_CLASSES), dtype='int')
	total_class_count = class_count.copy()
	for filename in tqdm(os.listdir(images_path)[:NUM_IMAGES]):
		try:
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
					iou, coords = IoU(bbox,box,return_coords=True)
					coords = (np.array(coords) - np.array([x1,y1,x1,y1])) / np.array([x2-x1,y2-y1,x2-x1,y2-y1])
					coords = (np.round(coords,3)*1000).astype('int')
					
					max_iou = max(max_iou,iou)
					if iou > 0.8 and class_count[class_index]<LIMIT_CLASS_IMAGES:
						box_write(filename, bbox, class_index, coords, image.shape[:2])
						class_count[class_index] += 1
						total_class_count[class_index] += 1
				
				if max_iou < 0.3 and class_count[0]<LIMIT_CLASS_IMAGES:
					box_write(filename, bbox, 0, (0,0,1000,1000), image.shape[:2])
					class_count[0] += 1
					total_class_count[0] += 1
		except Exception as e:
			print(f"#>#> Error occured in {filename}:\n{e}\n")
	save_rois(out_path)

if __name__=="__main__":
	main()