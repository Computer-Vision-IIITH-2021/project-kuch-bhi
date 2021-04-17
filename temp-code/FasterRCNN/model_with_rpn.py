import torchvision.models as models
import torch.nn as nn
import torch

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, x):
		return x

def initialize_model(NUM_CLASSES):
	model = models.vgg16(pretrained=True)

	# input size = 512x7x7, n=num_anchors=9
	n = 9
	model.rpn = nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
		nn.ReLU(inplace=True),
		nn.Conv2d(in_channels=256, out_channels=(2*n+4*n), kernel_size=1, padding=0),
		nn.ReLU(inplace=True)
	)

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

	return model