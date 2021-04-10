from torchvision.models import mobilenet_v2
import torch.nn as nn

def get_features(pretrained=True):
	# input: torch.Size([1, 3, 224, 224])
	# output: torch.Size([1, 32, 28, 28])
	model = mobilenet_v2(pretrained=pretrained)
	return model.features[:7]