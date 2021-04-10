from selective_search import find_region_proposals
import cv2
import numpy as np
from model import initialize_model, load_model
from dataset import prepare_image
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm

NUM_CLASSES = 21
image_size = (224,224)
MODEL_PATH = './checkpoints/regression 12-03-2021 22-38-46 epoch-2.pth'
model = initialize_model(NUM_CLASSES)
model, _, _ = load_model(model, MODEL_PATH)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(DEVICE)
print("Model Loaded...")
for param in model.parameters():
	param.requires_grad = False
torch.no_grad()
class_list = ['background','person','bird','cat','cow','dog','horse','sheep','aeroplane','bicycle','boat','bus','car','motorbike','train','bottle','chair','diningtable','pottedplant','sofa','tvmonitor']

def draw_boxes(image, boxes):
	new_image = image.copy()
	for box in boxes:
		class_ind = int(box[0])
		x1,y1,x2,y2 = tuple(map(int, box[1:]))
		new_image = cv2.rectangle(new_image, (x1,y1), (x2,y2), (255,0,0), 3)
		cv2.putText(new_image, class_list[class_ind], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
	return new_image

torch_hstack = lambda a: torch.cat(a, dim=1)

def predict(image, batch_size=4, nms_iou_threshold=0.7):
	bboxes = find_region_proposals(image)
	print(len(bboxes), "proposals found")

	all_class_probs = []
	all_box_preds = []

	for i in tqdm(range(0,len(bboxes),batch_size)):
	# for i in tqdm(range(500,521,batch_size)):
		current_bboxes = bboxes[i:i+batch_size]
		current_inputs = []
		for j in current_bboxes:
			x1,y1,x2,y2 = j
			current_inputs.append(prepare_image(cv2.resize(image[y1:y2,x1:x2], image_size)).unsqueeze(0))
		current_bboxes = torch.tensor(current_bboxes.astype('int32'))
		current_inputs = torch.cat(current_inputs).to(DEVICE)
		features = model.features(current_inputs).reshape(len(current_bboxes),-1)
		class_probs = model.classifier(features).cpu()
		box_preds = model.regressor(features).cpu()
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
	return final_boxes

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