import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../Images/airplane.png"
image = Image.open(image_path)
image = image.convert('RGB')  #THIS LINE  is only for PNG.
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Kaiyuan(nn.Module):
    def __init__(self):
        super(Kaiyuan, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("kaiyuan_29_gpu.pth")
print(model)
image = torch.reshape(image, (1,3,32,32))
image = image.cuda()
model.eval()
#transfrom to test type
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))