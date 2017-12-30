#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.append('/home/liang/caffe/python') #for caffe
import caffe
sys.path.append('/home/liang/anaconda2/lib') #for cv2
import cv2
import numpy as np
from os.path import join, isfile
import random



class ImageLabelmapDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.root=params['root']
        self.source=params['source']
        self.mean = np.array(params['mean'],dtype=np.float32)
        self.shuffle = bool(params['shuffle'])
        with open(join(self.root,self.source),'r') as f:
            self.filelist=f.readlines()
        if self.shuffle:
            random.shuffle(self.filelist)
        self.idx=0
        top[0].reshape(1,3,100,100)
        top[1].reshape(1,1,100,100)
    def reshape(self,bottom,top):
        "reshape in forward"
        pass
    def forward(self,bottom,top):
        [imgfile,lbfile]=self.filelist[self.idx].split()
        [imgfile,lbfile]=join(self.root,imgfile),join(self.root,lbfile)
        img=cv2.imread(imgfile).astype(np.float32)
        lb=cv2.imread(lbfile,0).astype(np.float32) #灰度图
        if img.ndim==2:
            img=img[:,:,np.newaxis]
            img=np.repeat(img,3,2)
        img=img-self.mean
        img=np.transpose(img,(2,0,1))
        img=img[np.newaxis,:,:,:]
        if lb.ndim!=2:
            raise Exception('lable shape error')
        if(lb.shape[0]!=img.shape[2] or lb.shape[1]!=img.shape[3]):
            raise Exception('label and image shape error')
        for i in range(lb.shape[0]):
            for j in range(lb.shape[1]):
                if lb[i,j]>255*0.5:
                    lb[i,j]=1
                elif lb[i,j]!=0:
                    lb[i,j]/=255.0
                else:
                    lb[i,j]=0
	#print(np.count_nonzero(lb))

        lb=lb[np.newaxis,np.newaxis,:,:]
        top[0].reshape(1,3,img.shape[2],img.shape[3])
        top[1].reshape(1,1,lb.shape[2],lb.shape[3])
        top[0].data[...]=img
        top[1].data[...]=lb
        if self.idx == len(self.filelist)-1:
            print("Restarting data prefetcing from start.")
            random.shuffle(self.filelist)
            self.idx=0
        else:
            self.idx=self.idx+1
    def backward(self,top,propagate_down,bottom):
        #no need
        pass
        
