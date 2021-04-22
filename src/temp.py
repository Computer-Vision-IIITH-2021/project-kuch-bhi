import resnet50_features
import vgg16_features
import mobilenet_v2_features
from torchsummary import summary
import torch

resnetfeatures = resnet50_features.get_features()
vgg16features = vgg16_features.get_features()
mobilenet_v2features = mobilenet_v2_features.get_features()

inp = torch.zeros((1,3,224,224))