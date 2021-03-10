import torchvision.models as models
from dataset import MyCustomDataset

DATA_PATH = '../../../training_data'

model = models.vgg16(pretrained=True)
train_set = MyCustomDataset(data_dir=DATA_PATH,split="train")
test_set = MyCustomDataset(data_dir=DATA_PATH,split="test")

