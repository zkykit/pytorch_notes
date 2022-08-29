import torch
import torchvision
from torch import nn
from torch.nn import Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

#from model import *

#define thed device for training
device = torch.device("cuda")

#prepare dataset
train_data = torchvision.datasets.CIFAR10("../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

#length
train_data_size = len(train_data)
test_data_size = len(test_data)

print("The length of train dataset is: {}".format(train_data_size))
print("The length of test dataset is: {}".format(test_data_size))

#use dataloader to load dataset
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


#make network model
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

kaiyuan = Kaiyuan()
kaiyuan = kaiyuan.to(device)

#loss function
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

#optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(kaiyuan.parameters(), lr=learning_rate)

#set parameters for train network
#record times of train
total_train_step = 0

#record times of test
total_test_step = 0

#epoch of train
epoch = 30

#add tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("-------------the {} times train begin----------".format(i+1))

    # train step begin
    for data in train_dataloader:
        #loss_network
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = kaiyuan(imgs)
        loss = loss_fn(outputs,targets)

        #optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            end_time = time.time()
            print(end_time-start_time)
            print("times of train: {}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # test step begin
    kaiyuan.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            #loss_network
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = kaiyuan(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_step + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy


    print("totel loss in testdataset is : {}".format(total_test_loss))
    print("accuracy of the total testdataset is : {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(kaiyuan,"kaiyuan_{}_gpu.pth".format(i))
    print("#############model has been saved###########")



writer.close()