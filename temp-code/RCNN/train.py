from torch.hub import import_module
import torchvision.models as models
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
img_size = (224,224)
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

if not os.path.exists("checkpoints"):
	os.mkdir("checkpoints")

model = models.vgg16(pretrained=True)

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, x):
		return x

model.avgpool = Identity()
model.classifier = nn.Sequential(
	nn.Linear(in_features=25088, out_features=4096, bias=True),
	nn.ReLU(inplace=True),
	nn.Dropout(p=0.5, inplace=False),
	nn.Linear(in_features=4096, out_features=4096, bias=True),
	nn.ReLU(inplace=True),
	nn.Dropout(p=0.5, inplace=False),
	nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True),
)

model.regressor = nn.Sequential(
	nn.Linear(in_features=25088, out_features=4096, bias=True),
	nn.ReLU(inplace=True),
	nn.Dropout(p=0.5, inplace=False),
	nn.Linear(in_features=4096, out_features=4096, bias=True),
	nn.ReLU(inplace=True),
	nn.Dropout(p=0.5, inplace=False),
	nn.Linear(in_features=4096, out_features=4, bias=True),
)

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

for param in model.classifier.parameters():
	param.requires_grad = True

optimizer = optim.Adam(model.classifier.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
criterion = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
	train_loss = 0
	with tqdm(train_loader, unit="batch", desc="Training", position=1, disable=False) as tepoch:
		for i,(imgs,labels) in enumerate(tepoch):
			imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).long()
			optimizer.zero_grad()
			preds = model.classifier(model.features(imgs).reshape(BATCH_SIZE,-1))
			loss = criterion(preds, labels)
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			tepoch.set_postfix(loss=train_loss/(i+1))
	PATH = os.path.join("checkpoints/", f'classification {currentTime()} epoch-{epoch+1}.pth')
	torch.save(model.state_dict(), PATH)

# ------------- ------------- ------------- ----
# ------------- Training Regressor -------------
# ------------- ------------- ------------- ----

# for param in model.regressor.parameters():
# 	param.requires_grad = True

# optimizer = optim.Adam(model.regressor.parameters(),lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
# criterion = nn.MSELoss()

# for epoch in range(NUM_EPOCHS):
# 	train_loss = 0
# 	with tqdm(train_loader, unit="batch", desc="Training", position=1, disable=False) as tepoch:
# 		for i,(imgs,labels) in enumerate(tepoch):
# 			imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
# 			optimizer.zero_grad()
# 			preds = model.classifier(model.features(imgs).reshape(BATCH_SIZE,-1))
# 			loss = criterion(preds, labels)
# 			loss.backward()
# 			optimizer.step()
# 			train_loss += loss.item()
# 			tepoch.set_postfix(loss=train_loss/(i+1))
# 	PATH = os.path.join("checkpoints/", f'classification {currentTime()} epoch-{epoch+1}.pth')
# 	torch.save(model.state_dict(), PATH)
