import torch
import torchvision
from torch import nn
from model_save import *
#method1 , load model
model = torch.load("vgg16_method1.pth")
#print(model)


#method2 , load model
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
#model = torch.load("vgg16_method2.pth")
#print(vgg16)
'''
class Kaiyuan(nn.Module):
    def __init__(self):
        super(Kaiyuan, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x
'''


model = torch.load("kaiyuan_method1.pth")
print(model)