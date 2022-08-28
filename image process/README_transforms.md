# transform image preprocess

## resize
```
	img = Image.open("path.jpg")
	trans_size = transforms.Resize((512,512))
	img_resize = trans_resize(img)
```

	compose(used for combine some steps)
```
	trans_compose = transforms.Compose(,)

```

	randomcrop （random resize and it can make more data，数据增强）
```
	transforms.RandomCrop((100,56)) 
	Should be attention, the random size should smaller than origin image size
```

## dataset use transform
	import
```
	import torchvision
```

```
	dataset_transfrom = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
	train_set = torchvision.datasets.datasetname(root="./dataset",train=True,transform=dataset_transfrom,download=True)

	The other part pls see py
```


