from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

# use transforms.ToTensor to see 2 questions
# 1. how to use transforms
# 2. why we nees Tensor datatype


# abs path : /home/zky/Desktop/learn_pytorch/dataset/train/ants_image/0013035.jpg
# rela path: dataset/train/ants_image/0013035.jpg
img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img",tensor_img)
writer.close()