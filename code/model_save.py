import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

#method of saving 1 (structure+parameter)
torch.save(vgg16,"vgg16_method1.pth")

#method of saving 2 (parameter) official recommend
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#xianjing :trap
class Kaiyuan(nn.Module):
    def __init__(self):
        super(Kaiyuan, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x
torchvision.models.vgg16(pretrained=False)