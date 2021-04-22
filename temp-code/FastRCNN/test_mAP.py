from predict import *
from xmlparse import parseXML
import os
from get_roi_boxes import IoU

images_path = "D:/VOCdevkit/VOC2012/JPEGImages/"
annots_path = "D:/VOCdevkit/VOC2012/Annotations/"
def main():
	mAP = 0
	predictions = {}
	for i in range(len(class_list)):
		predictions[i] = {'tp':0,'fp':0,'tn':0,'fn':0}
	for image_name in os.listdir(images_path)[10000:10000+1]:
		image_path = os.path.join(images_path, image_name)
		annot_path = os.path.join(annots_path, image_name.split('.')[0]+'.xml')
		image = cv2.imread(image_path, cv2.IMREAD_COLOR)
		boxes = predict(image)
		actual_classes, actual_boxes = parseXML(annot_path)
		for actual_class, actual_box in zip(actual_classes, actual_boxes):
			found_actual = False
			actual_ind = class_list.index(actual_class)
			for b in boxes:
				found_pred = False
				class_ind = b[0]
				pred_box = b[1:]
				if IoU(actual_box, pred_box, return_coords=True)[0] > 0.5:
					found_pred = True
					if (class_ind==actual_ind):
						found_actual = True
						predictions[actual_ind]['tp'] += 1
						for c in range(len(class_list)):
							if c!=actual_ind:
								predictions[c]['tn'] += 1
				if not found_pred:
					predictions[class_ind]['fp'] += 1
			if not found_actual:
				predictions[class_ind]['fn'] += 1

	for i in predictions:
		tp,fp,fn,tn = predictions[i]['tp'], predictions[i]['fp'], predictions[i]['fn'], predictions[i]['tn']
		precision = tp/(tp+fp+1e-10)
		recall = tp/(tp+fn+1e-10)
		mAP += precision
	print('mAP:', mAP/len(class_list))

if __name__=='__main__':
	main()