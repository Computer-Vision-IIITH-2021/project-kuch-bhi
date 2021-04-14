from selective_search import find_region_proposals
import cv2
import numpy as np
from roi_pooling import roi_pooling
from model import Network, load_model
from dataset import prepare_image
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

MODEL_NAME = 'vgg16'
NUM_CLASSES = 21
image_size = (224,224)
MODEL_PATH = './checkpoints/classification 14-04-2021 19-40-43 epoch-2.pth'
model = Network(MODEL_NAME, NUM_CLASSES)
model, _, _ = load_model(model, MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE)
print("Model Loaded...")
for param in model.parameters():
	param.requires_grad = False
torch.no_grad()
class_list = ['background','person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','diningtable','pottedplant','sofa','tvmonitor']

def batch_to_rois2(batch):
	imgs, labels = batch
	imgs = imgs.to(DEVICE)
	roi_batch, indices = roi_pooling(model.features(imgs), labels[:,:,:4])
	roi_batch = roi_batch.to(DEVICE)
	b,n = labels.shape[:2]
	roi_labels = labels[:,:,:4].reshape(b*n,4)[indices].type(torch.float32).to(DEVICE)
	return roi_batch, roi_labels

def draw_boxes(image, boxes):
	new_image = image.copy()
	for box in boxes:
		class_ind = int(box[0])
		x1,y1,x2,y2 = tuple(map(int, box[1:]))
		new_image = cv2.rectangle(new_image, (x1,y1), (x2,y2), (255,0,0), 3)
		cv2.putText(new_image, class_list[class_ind], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
	return new_image

torch_hstack = lambda a: torch.cat(a, dim=1)

def predict(image, batch_size=16, nms_iou_threshold=0.7):
	bboxes = find_region_proposals(image).astype('float')
	# np.save('temp_bboxes.npy', bboxes)
	# bboxes = np.load('temp_bboxes.npy')
	H,W = image.shape[:2]
	bboxes[:,0] /= W
	bboxes[:,1] /= H
	bboxes[:,2] /= W
	bboxes[:,3] /= H
	print(len(bboxes), "proposals found")

	img_ready = prepare_image(cv2.resize(image, image_size)).unsqueeze(0).to(DEVICE)
	print(img_ready.shape)
	roi_batch, roi_bboxes = batch_to_rois2((img_ready, torch.tensor([bboxes])))
	print("Proposals left:", len(roi_bboxes), "==", len(roi_batch))
	all_class_probs = []
	all_box_preds = []

	roi_bboxes[:,0] *= W
	roi_bboxes[:,1] *= H
	roi_bboxes[:,2] *= W
	roi_bboxes[:,3] *= H

	for i in tqdm(range(0,len(roi_bboxes),batch_size)):
	# for i in tqdm(range(500,521,batch_size)):
		current_bboxes = roi_bboxes[i:i+batch_size]
		current_bboxes = current_bboxes.type(torch.int32).cpu()
		class_probs = model.classifier(roi_batch[i:i+batch_size]).cpu()
		box_preds = model.regressor(roi_batch[i:i+batch_size]).cpu()
		x1,y1,x2,y2 = [current_bboxes[:,[i]] for i in range(4)]
		box_preds *= torch_hstack([x2-x1,y2-y1,x2-x1,y2-y1]).type(torch.float32)
		box_preds = box_preds.long() + torch_hstack([x1,y1,x1,y1]).long()
		all_class_probs.append(class_probs)
		all_box_preds.append(box_preds)
	all_class_probs = torch.cat(all_class_probs)
	all_box_preds = torch.cat(all_box_preds)

	final_boxes = []
	class_preds = all_class_probs.argmax(axis=1)

	boxes_to_keep = torchvision.ops.boxes.batched_nms(all_box_preds.type(torch.float32), all_class_probs, class_preds, nms_iou_threshold)
	final_boxes = torch_hstack([class_preds[boxes_to_keep].reshape(-1,1),all_box_preds[boxes_to_keep]])
	return final_boxes.detach().cpu().numpy()

def main():
	image_path = r"C:\Users\trizo\Downloads\Documents\Sem6\CV\Project\VOC2012\JPEGImages\2007_000027.jpg"
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	boxes = predict(image)
	predicted = cv2.cvtColor(draw_boxes(image, boxes), cv2.COLOR_BGR2RGB)
	plt.axis('off')
	plt.imshow(predicted)
	plt.show()

if __name__=='__main__':
	main()