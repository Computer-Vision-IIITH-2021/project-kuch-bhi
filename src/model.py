import torch.nn as nn
import torch
import resnet50_features
import vgg16_features
import mobilenet_v2_features

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, x):
		return x

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
		pass
	def forward(self, x):
		return x.view(x.shape[0],-1)

class Network(nn.Module):
	def __init__(self, model_name='vgg16', NUM_CLASSES=21):
		super(Network, self).__init__()
		features_final_channels = {'vgg16':256, 'resnet50':512, 'mobilenet_v2':32}
		if model_name=='vgg16':
			self.features = vgg16_features.get_features()
		elif model_name=='resnet50':
			self.features = resnet50_features.get_features()
		elif model_name=='mobilenet_v2':
			self.features = mobilenet_v2_features.get_features()

		self.classifier = nn.Sequential(
			nn.Conv2d(in_channels=features_final_channels[model_name], out_channels=512, kernel_size=1, padding=0),
			Flatten(),
			nn.Linear(in_features=512*7*7, out_features=4096, bias=True),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5, inplace=False),
			nn.Linear(in_features=4096, out_features=4096, bias=True),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5, inplace=False),
			nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True),
		)

		self.regressor = nn.Sequential(
			nn.Conv2d(in_channels=features_final_channels[model_name], out_channels=512, kernel_size=1, padding=0),
			Flatten(),
			nn.Linear(in_features=512*7*7, out_features=4096, bias=True),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5, inplace=False),
			nn.Linear(in_features=4096, out_features=4096, bias=True),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5, inplace=False),
			nn.Linear(in_features=4096, out_features=4, bias=True),
		)

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