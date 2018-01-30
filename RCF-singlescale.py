
# coding: utf-8

# In[1]:

# Make sure that caffe is on the python path:
caffe_root = '/home/liang/rcf/'  # this file is expected to be in {caffe_root}/examples/hed/
import sys
sys.path.insert(0, '/home/liang/caffe/python/')
sys.path.insert(0, '/home/liang/anaconda2/lib/')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import cv2
import time




import caffe


# In[2]:

data_root = '/home/liang/caffe/data/HED-BSDS/HED-BSDS/'
with open(data_root+'test.lst') as f:
    test_lst = f.readlines()
    
test_lst = [x.strip() for x in test_lst]


# In[3]:

im_lst = []
for i in range(0, len(test_lst)):
    im = Image.open(data_root+test_lst[i])
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    im_lst.append(in_)


# In[4]:

#Visualization
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size/2
    
    plt.figure()
    for i in range(0, len(scale_lst)):
        s=plt.subplot(1,5,i+1)
        plt.imshow(1-scale_lst[i], cmap = cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()


# In[ ]:

#remove the following two lines if testing with cpu
caffe.set_mode_gpu()
caffe.set_device(0)
# load net
net = caffe.Net('hed_test.prototxt', 'hed_bsds_iter_18000.caffemodel', caffe.TEST)



# In[6]:

save_root = os.path.join(data_root, 'test-fcn1')
if not os.path.exists(save_root):
    os.mkdir(save_root)
start_time = time.time()
for idx in range(0, len(test_lst)):
    in_ = im_lst[idx]
    in_ = in_.transpose((2, 0, 1))
    print(in_.shape)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    
    # save results
    #out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
    #out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
    #out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
    #out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
    #out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
    fuse = net.blobs['sigmoid_fuss'].data[0][0, :, :]
    #out1 = 255 * (1-out1)
    #out2 = 255 * (1-out2)
    #out3 = 255 * (1-out3)
    #out4 = 255 * (1-out4)
    #out5 = 255 * (1-out5)
    fuse = 255 * (1-fuse)
    #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out1.png', out1)
    #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out2.png', out2)
    #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out3.png', out3)
    #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out4.png', out4)
    #cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_out5.png', out5)
    cv2.imwrite(save_root + '/' + test_lst[idx][5:-4] + '_fuse.png', fuse)
    
diff_time = time.time() - start_time
print 'Detection took {:.3f}s per image'.format(diff_time/len(test_lst))


# In[8]:



# In[9]:




# In[ ]:



