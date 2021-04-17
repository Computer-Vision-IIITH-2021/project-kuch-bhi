from model import *
from dataset import MyCustomDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
currentTime = lambda : datetime.now().strftime("%d-%m-%Y %H-%M-%S")

DATA_PATH = '../../../training_data'
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 0
NUM_CLASSES = 21
torch.manual_seed(SEED)
image_size = (224,224)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
ITERS_TO_TEST = 50
TRAIN_CLASSIFIER = False
TRAIN_REGRESSOR = True
CHECKPOINT = 'checkpoints/classification 12-03-2021 22-31-17 epoch-2.pth'

if not os.path.exists("checkpoints"):
	os.mkdir("checkpoints")

model = initialize_model(NUM_CLASSES)

model = model.to(DEVICE)
for param in model.parameters():
	param.requires_grad = False

train_set = MyCustomDataset(data_dir=DATA_PATH,split="train")
test_set = MyCustomDataset(data_dir=DATA_PATH,split="test")
train_loader = DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=False)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=False)

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
	for epoch in range(NUM_EPOCHS):
		train_loss = 0
		num_correct_train = 0
		num_total_train = 0
		with tqdm(train_loader, unit="batch", desc=f"Training Classifier Epoch {epoch+1}", position=1, disable=False) as tepoch:
			for i,(imgs,labels) in enumerate(tepoch):
				imgs, labels = imgs.to(DEVICE), labels[:,0].to(DEVICE).long()
				optimizer.zero_grad()
				preds = model.classifier(model.features(imgs).reshape(imgs.shape[0],-1))
				loss = criterion(preds, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				num_correct_train += find_num_correct(preds, labels)
				num_total_train += len(labels)
				tepoch.set_postfix(loss=train_loss/(i+1), acc=num_correct_train/num_total_train)
				iters += 1
				if iters % ITERS_TO_TEST == 0:
					with torch.no_grad():
						test_loss = 0
						num_correct_test = 0
						num_total_test = 0
						with tqdm(test_loader, unit="batch", desc=f"Validation (Classifier)", position=2, disable=False) as valepoch:
							for j,(imgs,labels) in enumerate(valepoch):
								imgs, labels = imgs.to(DEVICE), labels[:,0].to(DEVICE).long()
								preds = model.classifier(model.features(imgs).reshape(imgs.shape[0],-1))
								loss = criterion(preds, labels)
								test_loss += loss.item()
								num_correct_test += find_num_correct(preds, labels)
								num_total_test += len(labels)
								valepoch.set_postfix(loss=test_loss/(j+1), acc=num_correct_test/num_total_test)

		PATH = os.path.join("checkpoints/", f'classification {currentTime()} epoch-{epoch+1}.pth')
		# torch.save(model.state_dict(), PATH)
		save_model(model, PATH, classifier_optimizer=optimizer)



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
	for epoch in range(NUM_EPOCHS):
		train_loss = 0
		with tqdm(train_loader, unit="batch", desc=f"Training Regressor Epoch {epoch+1}", position=1, disable=False) as tepoch:
			for i,(imgs,labels) in enumerate(tepoch):
				imgs, labels = imgs.to(DEVICE), labels[:,1:].to(DEVICE).type(torch.float32)
				optimizer.zero_grad()
				preds = model.regressor(model.features(imgs).reshape(imgs.shape[0],-1))
				loss = criterion(preds, labels)
				loss.backward()
				optimizer.step()
				train_loss += loss.item()
				tepoch.set_postfix(loss=train_loss/(i+1))
				iters += 1
				if iters % ITERS_TO_TEST == 0:
					with torch.no_grad():
						test_loss = 0
						with tqdm(test_loader, unit="batch", desc=f"Validation (Regressor)", position=2, disable=False) as valepoch:
							for j,(imgs,labels) in enumerate(valepoch):
								imgs, labels = imgs.to(DEVICE), labels[:,1:].to(DEVICE).type(torch.float32)
								preds = model.regressor(model.features(imgs).reshape(imgs.shape[0],-1))
								loss = criterion(preds, labels)
								test_loss += loss.item()
								valepoch.set_postfix(loss=test_loss/(j+1))

		PATH = os.path.join("checkpoints/", f'regression {currentTime()} epoch-{epoch+1}.pth')
		save_model(model, PATH, regressor_optimizer=optimizer)
