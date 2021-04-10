import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

def prepare_image(t):
	t = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)
	t = t.astype('float32')/255
	t = t.transpose(2,0,1)
	return torch.from_numpy(t)

class MyCustomDataset(Dataset):
	def __init__(self, data_dir, split="train", test_ratio=0.15, shuffle=True, seed=0):
		self.data_dir = data_dir
		img_list = os.listdir(data_dir)
		np.random.seed(seed)
		if shuffle:
			np.random.shuffle(img_list)
		cut_point = int(len(img_list)*(1-test_ratio))
		if split == "train":
			self.img_list = img_list[:cut_point]
		else:
			self.img_list = img_list[cut_point:]

	def __getitem__(self, index):
		filename = self.img_list[index]
		label = list(map(int, filename.split('_')[:-1]))
		label = torch.tensor(label)
		label[1:] /= 1000
		img_path = os.path.join(self.data_dir, filename)
		img = cv2.imread(img_path, cv2.IMREAD_COLOR)
		input_img = prepare_image(img)
		return (input_img, label)

	def __len__(self):
		return len(self.img_list)