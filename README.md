# object-detection-tutorial
This is a simple object detection tutorial. In this tutorial, you will learn some basic knowledge and technology about object detection. In addition, I have achieved a few simple OD models whose style are close to YOLO(You Only Look Once).

For researchers in China, you can learn this tutorial with my column in zhihu:

https://zhuanlan.zhihu.com/p/91843052


## Installation
- My PyTorch version is 1.1.0.
- My anaconda3 version is 5.1.0 whose python is 3.6
- You'd better to install tensorboard > 1.13.
- For testing, I use opencv in python to visualize the detection results.
So, you also need to install python-opencv.

## Dataset
Because this is just a tutorial, I really do not want to download MSCOCO datatset. Just PASCAL VOC2007 and 2012. 

### VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

#### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

### Train
- For example, if you want to train yolo-v1 designed by myself:

```Shell
python train_voc.py
```

if yolo-anchor:
```Shell
python train_voc.py -v yolo_anchor
```

The default backbone network is resnet18. If you want to use darknet19:

```Shell
python train_voc.py -bk d19
```

The model will be saved in `weights/` by default.

### Test

For example, you want to test the yolo-v1 model on VOC2007 test:

```Shell
python test_voc.py --trained_model [ Please write down your trained model dir. ]
```

### Evaluation
For example, you want to evaluate the yolo-v1 model on VOC2007 test:

```Shell
python eval_voc.py --trained_model [ Please write down your trained model dir. ]
```

### Train your own dataset
First, you need to make a VOC-style dataset. The names of your images are as following( .png or .jpg or whichever image format you like ):

```Shell
000000.png, 000001.png, 000002.png, ...
```
And your annotations' names are as: 

```Shell
000000.xml, 000001.xml, 000002.xml, ...
```

Then, you need to write your dataloader file. You can modify `data/voc0712.py` to fit your dataset. What you need to do is just to change the path to images and annotations.

Finally, please modify the `train_voc.py`.