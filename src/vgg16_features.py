from torchvision.models import vgg16
import torch.nn as nn

def get_features(pretrained=True):
	# input: torch.Size([1, 3, 224, 224])
	# output: torch.Size([1, 256, 28, 28])
	model = vgg16(pretrained=pretrained)
	return model.features[:17]