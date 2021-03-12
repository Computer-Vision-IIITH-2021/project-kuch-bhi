from re import I
from selective_search import find_region_proposals
import cv2
import numpy as np
from model import initialize_model, load_model
from dataset import prepare_image
import torch
import matplotlib.pyplot as plt
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

def predict(image_path, batch_size=4):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	new_image = image.copy()
	bboxes = find_region_proposals(image)
	print(len(bboxes), "proposals found")
	images = []
	boxes = []
	for i, bbox in tqdm(enumerate(bboxes)):
		x1,y1,x2,y2 = bbox
		img = prepare_image(cv2.resize(image[y1:y2,x1:x2], image_size)).unsqueeze(0).to(DEVICE)
		class_preds = model.classifier(model.features(img).reshape(1,-1)).argmax(axis=1).cpu()
		box_preds = model.regressor(model.features(img).reshape(1,-1)).cpu()
		box_preds *= torch.tensor([x2-x1,y2-y1,x2-x1,y2-y1], dtype=torch.float32)
		box_preds = box_preds.long() + torch.tensor([x1,y1,x1,y1], dtype=torch.long)
		class_ind = class_preds[0].item()
		if class_ind != 0:
			box_preds = tuple(map(int, box_preds[0]))
			new_image = cv2.rectangle(new_image, box_preds[:2], box_preds[2:], (255,0,0), 3)
			cv2.putText(new_image, class_list[class_ind], box_preds[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
		# boxes.append(bbox)
		# images.append(img)
		# if (i+1)%batch_size == 0:
		# 	images = torch.tensor(images)
		# 	boxes = torch.tensor
		# 	class_preds = model.classifier(model.features(images).reshape(images.shape[0],-1)).argmax(axis=1)
		# 	box_preds = model.regressor(model.features(images).reshape(images.shape[0],-1))
		# 	box_preds *= torch.tensor([x2-x1,y2-y1,x2-x1,y2-y1], dtype=torch.float32)
		# 	box_preds = box_preds.long() + torch.tensor([x1,y1,x1,y1], dtype=torch.long)
		# 	cv2.rectangle(new_image, (x1,y1))

	return new_image

def main():
	image_path = r"C:\Users\trizo\Downloads\Documents\Sem6\CV\Project\VOC2012\JPEGImages\2007_000027.jpg"
	predicted = cv2.cvtColor(predict(image_path), cv2.COLOR_BGR2RGB)
	plt.axis('off')
	plt.imshow(predicted)
	plt.show()

main()