## Preview

My graduation project is edge detection. My basic task is to understand and build rcf. Finally improve it.

## Background

### rcf
      paper: https://arxiv.org/abs/1612.02103 
      code: https://github.com/yun-liu/rcf   
   
#### rcf is based on hed.

### hed 
      paper: https://arxiv.org/abs/1504.06375 
      code: https://github.com/s9xie/hed      

#### a pycaffe code for hed and rcf : https://github.com/zeakey/hed

2017/12/30

caffe provides image_label_layer for image-to-label classification, such as mnist, cifar, iamgenet,etc. In edge detection, this task is for image-to-image, there is not available data layer for image-to-image, so we need to add a layer for it.
There are two ways for us to add a layer in caffe. Cpp and python layer.
1. cpp: http://blog.csdn.net/wfei101/article/details/76735760
The image_labelmap_data_layer in hed and rcf use this method. However, this layer uses many opencv methods, some methods are difficult to understand.
2. python layer:http://blog.csdn.net/liuheng0111/article/details/53090473
Python layer is much more easier than cpp layer.Especially for data layer.We can use cv2 to load images.And there are some examples for reference.
1) FCN: VOCSegDataLayer
https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc_layers.py
2) caffe/examples/pycaffe/layers/pyloss.py: EuclideanLossLayer
https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pyloss.py
Zeakey's ImageLabelmapDataLayer is written by python. So i decide to follow his way to use python to write this data layer.
ImageLabelmapdata.py 
image:1,3,h,w
lb:1,1,h,w










