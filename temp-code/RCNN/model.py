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

def save_model(model, filepath, classifier_optimizer=None, regressor_optimizer=None):
	cosd = None	if classifier_optimizer is None else classifier_optimizer.state_dict()
	rosd = None	if regressor_optimizer is None else regressor_optimizer.state_dict()
	
	torch.save({
		'model_state_dict': model.state_dict(),
		'classifier_optimizer_state_dict': cosd,
		'regressor_optimizer_state_dict': rosd}, filepath)

def load_model(model, filepath, classifier_optimizer=None, regressor_optimizer=None):
	checkpoint = torch.load(filepath)
	model.load_state_dict(checkpoint['model_state_dict'])
	cosd = checkpoint['classifier_optimizer_state_dict']
	rosd = checkpoint['regressor_optimizer_state_dict']
	if cosd is not None and classifier_optimizer is not None:
		classifier_optimizer.load_state_dict(cosd)
	if rosd is not None and regressor_optimizer is not None:
		regressor_optimizer.load_state_dict(rosd)
	return model, classifier_optimizer, regressor_optimizer