import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import json

roi_dict = {}
with open('../../../training_rois.json', 'r') as f:
	roi_dict = json.load(f)

str_to_list = lambda strf: list(map(int, strf.split('_')))
LIMIT_ROIs = 8

def prepare_image(t):
	t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
	t = cv2.resize(t, (224,224))
	t = t.astype('float32')/255
	t = t.transpose(2,0,1)
	return torch.from_numpy(t)

class Full_Images_Data(Dataset):
	def __init__(self, data_dir, split="train", test_ratio=0.15, shuffle=True, seed=0):
		self.data_dir = data_dir
		# img_list = os.listdir(data_dir)[:10]
		img_list = list(roi_dict.keys())
		np.random.seed(seed)
		if shuffle:
			np.random.shuffle(img_list)
		cut_point = int(len(img_list)*(1-test_ratio))
		self.img_list = img_list[:cut_point] if split == "train" else img_list[cut_point:]

	def __getitem__(self, index):
		filename = self.img_list[index]
		label = [str_to_list(i) for i in roi_dict[filename]][:LIMIT_ROIs]
		label = torch.tensor(label, dtype=torch.float32)
		label /= 1000
		label[:,4] *= 1000
		img_path = os.path.join(self.data_dir, filename)
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		input_img = prepare_image(img)
		return (input_img, label)

	def __len__(self):
		return len(self.img_list)