from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter("logs")
img = Image.open("Images/download (1).jpeg")
print(img)


#ToTensor 
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)


#normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([6,3,2],[9,3,5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("NOrmalize", img_norm,2)


#resize
print(img.size)
trans_resize = transforms.Resize((512,512))
#img PIL -> REsize -> img_resize PIL
img_resize = trans_resize(img)
print(img_resize)
print(type(img_resize))
#img_resize PIL -> totensor -> img_resize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize",img_resize,0)
print(img_resize)


#compose - resize - 2
trans_resize_2 = transforms.Resize(512)

#PIL ->tensor
trans_compose = transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize",img_resize_2,1)


# randomcrop
trans_random = transforms.RandomCrop((100,56))
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop= trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop,i)


writer.close()