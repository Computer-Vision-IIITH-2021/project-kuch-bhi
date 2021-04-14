from model import *
from dataset import Full_Images_Data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from roi_pooling import roi_pooling
currentTime = lambda : datetime.now().strftime("%d-%m-%Y %H-%M-%S")

torch.cuda.empty_cache()
DATA_PATH = '../../../VOC2012/JPEGImages'
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 0
NUM_CLASSES = 21
MODEL_NAME = 'vgg16'
torch.manual_seed(SEED)
image_size = (224,224)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
ITERS_TO_TEST = 50
MAX_STOP_COUNT = 3
TRAIN_CLASSIFIER = True
TRAIN_REGRESSOR = False
# CHECKPOINT = 'checkpoints/classification 12-03-2021 22-31-17 epoch-2.pth'
CHECKPOINT = None

if not os.path.exists("checkpoints"):
	os.mkdir("checkpoints")

model = Network(MODEL_NAME, NUM_CLASSES)

model = model.to(DEVICE)
for param in model.parameters():
	param.requires_grad = False

train_set = Full_Images_Data(data_dir=DATA_PATH,split="train")
test_set = Full_Images_Data(data_dir=DATA_PATH,split="test")
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=False)

def batch_to_rois(batch, mode='classfication'):
	imgs, labels = batch
	imgs = imgs.to(DEVICE)
	roi_batch, indices = roi_pooling(model.features(imgs), labels[:,:,:4])
	roi_batch = roi_batch.to(DEVICE)
	b,n = labels.shape[:2]
	if mode=='classfication':
		roi_labels = labels[:,:,4].reshape(b*n)[indices].long().to(DEVICE)
	else:
		roi_labels = labels[:,:,5:].reshape(b*n,4)[indices].type(torch.float32).to(DEVICE)
	return roi_batch, roi_labels

# ------------- ------------- ------------- -----
# ------------- Training Classifier -------------
# ------------- ------------- ------------- -----

def find_num_correct(preds, labels):
	return (torch.argmax(preds, axis=1)==labels).sum().item()

for param in model.parameters():
	param.requires_grad = False

if TRAIN_CLASSIFIER:
	for param in model.classifier.parameters():
		param.requires_grad = True

	optimizer = optim.Adam(model.classifier.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
	criterion = nn.CrossEntropyLoss()

	if CHECKPOINT is not None:
		model, optimizer, _ = load_model(model, CHECKPOINT, classifier_optimizer=optimizer)

	iters = 0
	min_test_loss = np.inf
	stop_count = 0
	for epoch in range(NUM_EPOCHS):
		train_loss = 0
		num_correct_train = 0
		num_total_train = 0
		with tqdm(train_loader, unit="batch", desc=f"Training Classifier Epoch {epoch+1}", position=1, disable=False) as tepoch:
			for i, batch in enumerate(tepoch):
				imgs, labels = batch_to_rois(batch)
				optimizer.zero_grad()
				preds = model.classifier(imgs)
				loss = criterion(preds, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				num_correct_train += find_num_correct(preds, labels)
				num_total_train += len(labels)
				tepoch.set_postfix(loss=train_loss/(i+1), acc=num_correct_train/num_total_train)
				iters += 1
		with torch.no_grad():
			test_loss = 0
			num_correct_test = 0
			num_total_test = 0
			with tqdm(test_loader, unit="batch", desc=f"Validation (Classifier)", position=2, disable=False) as valepoch:
				for j,batch in enumerate(valepoch):
					imgs, labels = batch_to_rois(batch)
					preds = model.classifier(imgs)
					loss = criterion(preds, labels)
					test_loss += loss.item()
					num_correct_test += find_num_correct(preds, labels)
					num_total_test += len(labels)
					valepoch.set_postfix(loss=test_loss/(j+1), acc=num_correct_test/num_total_test)
		if test_loss < min_test_loss:
			PATH = os.path.join("checkpoints/", f'classification {currentTime()} epoch-{epoch+1}.pth')
			save_model(model, PATH, regressor_optimizer=optimizer)
			min_test_loss = test_loss
		else:
			stop_count+=1
			if stop_count>=MAX_STOP_COUNT:
				break

# ------------- ------------- ------------- ----
# ------------- Training Regressor -------------
# ------------- ------------- ------------- ----

for param in model.parameters():
	param.requires_grad = False

if TRAIN_REGRESSOR:

	for param in model.regressor.parameters():
		param.requires_grad = True

	optimizer = optim.Adam(model.regressor.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
	criterion = nn.MSELoss()

	if CHECKPOINT is not None:
		model, _, optimizer = load_model(model, CHECKPOINT, regressor_optimizer=optimizer)

	iters = 0
	min_test_loss = np.inf
	stop_count = 0
	for epoch in range(NUM_EPOCHS):
		train_loss = 0
		with tqdm(train_loader, unit="batch", desc=f"Training Regressor Epoch {epoch+1}", position=1, disable=False) as tepoch:
			for i, batch in enumerate(tepoch):
				imgs, labels = batch_to_rois(batch, mode='regression')
				optimizer.zero_grad()
				preds = model.regressor(imgs)
				loss = criterion(preds, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				tepoch.set_postfix(loss=train_loss/(i+1))
				iters += 1
		with torch.no_grad():
			test_loss = 0
			with tqdm(test_loader, unit="batch", desc=f"Validation (Regressor)", position=2, disable=False) as valepoch:
				for j, batch in enumerate(valepoch):
					imgs, labels = batch_to_rois(batch, mode='regression')
					preds = model.regressor(imgs)
					loss = criterion(preds, labels)
					test_loss += loss.item()
					valepoch.set_postfix(loss=test_loss/(j+1))
		if test_loss < min_test_loss:
			PATH = os.path.join("checkpoints/", f'regression {currentTime()} epoch-{epoch+1}.pth')
			save_model(model, PATH, regressor_optimizer=optimizer)
			min_test_loss = test_loss
		else:
			stop_count+=1
			if stop_count>=MAX_STOP_COUNT:
				break
