import torchvision.models
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)

#print(vgg16_True)

train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

#print(vgg16_True)

vgg16_True.classifier.add_module("add_linear", nn.Linear(1000,10))
print(vgg16_True)
#vgg16_false.classifier[6] = nn.Linear(4096,10)
#print(vgg16_false)
