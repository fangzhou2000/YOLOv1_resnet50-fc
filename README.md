# YOLOv1
Pytorch implementation of YOLOv1

This implementation is pure resnet50 + fc, but its results is not so good, maybe there are bugs, I'll fix them later.

## Datasets
Download Pascal VOC 2007 trainval and test datasets and Pascal VOC 2012 trainval datasets to the following paths:
```
/home/username/datasets/VOC2007/VOCtrainval_06-Nov-2007/VOC2007
/home/username/datasets/VOC2007/VOCtest_06-Nov-2007/VOC2007
/home/username/datasets/VOC2012/VOCtrainval_11-May-2012/VOC2012
```
Change the path variables in the code, usually with the name "root".

## Train
```
cd YOLOv1
pip install requirement.txt
python train.py
```
The model uses pretrained ResNet50 as backbone and uses the same hyperparameters and training strategies following the original yolo.

I trained it on a single GTX 1080 Ti, it takes about one day. 

If you want to change the hyperparameters, please manually change relative codes.

## Test
The test results are as follows:

todo

## Acknowledgment
The implementation is largely refer to the repo[https://github.com/EclipseR33/yolo_v1_pytorch] and this repo is only used for learning.