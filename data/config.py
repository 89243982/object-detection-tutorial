# config.py
import os.path


# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
MEANS = (104, 117, 123)

# ANCHOR SIZE is searched by myself!!!
ANCHOR_SIZE = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]

# MULTI ANCHOR SIZE, we set the same number of anchor boxes in each scale.
IGNORE_THRESH = 0.5

# SIZE_THRESH is the critical point of anchor-free and anchor-based.
SIZE_THRESH = 0.05

# yolo-v1 config
voc_af = {
    'num_classes': 20,
    'lr_epoch': (150, 200, 250),
    'max_epoch': 250,
    'min_dim': [448, 448],
    'scale_thresh':[[0, 0.046], [0.046, 0.227], [0.227, 1.0]], 
    'clip': True,
    'name': 'VOC',
}

# yolo-anchor config
voc_ab = {
    'num_classes': 20,
    'lr_epoch': (60, 90, 160),
    'max_epoch': 160,
    'min_dim': [416, 416],
    'stride': 32,
    'strides': [16, 32],
    'clip': True,
    'name': 'VOC',
}

imagenet = {
    'batch_size': 64,
    'resize': 448,
    'max_epoch': 10,
    'lr': 1e-3,
    'data_path': "/home/k545/object-detection/myYOLO/backbone/imagenet/",
    'model_name': 'resnet18'
}