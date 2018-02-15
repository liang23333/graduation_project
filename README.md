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

## rcf
### train
      1、build caffe
      2、download the repository
         change caffe/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp to lib/sigmoid_cross_entropy_loss_layer.cpp in this code
         change caffe/include/caffe/sigmoid_cross_entropy_loss_layer.hpp to lib/sigmoid_cross_entropy_loss_layer.hpp in this code
         delete caffe/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cu
         open terminate,change the path to caffe,enter make caffe,make pycaffe
         download the 5stage-vgg.caffemodel from https://github.com/yun-liu/rcf or https://pan.baidu.com/s/1smhfKgX
         download data from https://github.com/yun-liu/rcf
      3、open solve.py ,change the caffe_root , change the 17th code to solver = caffe.SGDSolver('mysolver.prototxt')
      4、open terminate ,enter ./train.sh
### test
      1、build caffe
      2、download rcf_bsds_iter_18000.caffemodel from 
      3、use RCF-singlescale.py to test ，if you want to do edge detection on your image ,just edit data_root+'test.lst' ,and write it like what in it.
### results
      I get the ODS F-measure 0.787 less than 0.803 in https://github.com/yun-liu/rcf, because i train it use 2/3 BSDS data, my GPU is GT740M, 2G , not enough for 1.5 scale data.



## hed
### train
      1、build caffe
      2、download the repository
         change caffe/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp to lib/sigmoid_cross_entropy_loss_layer.cpp in this code
         change caffe/include/caffe/sigmoid_cross_entropy_loss_layer.hpp to lib/sigmoid_cross_entropy_loss_layer.hpp in this code
         delete caffe/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cu
         open terminate,change the path to caffe,enter make caffe,make pycaffe
         download the 5stage-vgg.caffemodel from https://github.com/yun-liu/rcf or https://pan.baidu.com/s/1smhfKgX
         download data from https://github.com/yun-liu/rcf
      3、open solve.py ,change the caffe_root
      4、open terminate ,enter ./train.sh
### test
      1、build caffe
      2、download hed_bsds_iter_18000.caffemodel from https://pan.baidu.com/s/1smhfKgX
      3、use RCF-singlescale.py to test ，if you want to do edge detection on your image ,just edit data_root+'test.lst' ,and write it like what in it.
### results
      I get the ODS F-measure 0.784 less than 0.790 in https://github.com/s9xie/hed, because i train it use 2/3 BSDS data, my GPU is GT740M, 2G , not enough for 1.5 scale data.


## question
      when i train the net ,the differences in the loss are always very large, both in the beginning and end of the training, but the test results are different , are there any errors in my process ,or some problems in the loss function ?If you have any ideas,please contact me by 137809083@qq.com.
      
      
      
