# Tensorboard

## Used for make images and datas

### some basic code 
	import command(import PIL or cv2 are the same.)
```
	from torch.utils.tensorboard import SummaryWriter
	from PIL import Image
```

	SummaryWriter
```
	writer = SummaryWriter("logs")
	writer.add_image("train",img_array,2,dataformats='HWC') 
	writer.close()
```

	use transforms
	import command
```
	from torchvision import transforms
```

```
	tensor_trans = trans.ToTensor()
	tensor_img = tensor_trans(imgs)
	writer.add_image("Tensor_img",tensor_img)
	writer.close()
```

## start at Terminal

```
	tensorboard --logdir= titlename
```




