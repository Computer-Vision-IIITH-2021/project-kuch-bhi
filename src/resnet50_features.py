from torchvision.models import resnet50
import torch.nn as nn

def get_features(pretrained=True):
	# input: torch.Size([1, 3, 224, 224])
	# output: torch.Size([1, 512, 28, 28])
	model = resnet50(pretrained=pretrained)
	layers = [
		model.conv1,
		model.bn1,
		model.relu,
		model.maxpool,
		model.layer1,
		model.layer2
	]
	return nn.Sequential(*layers)