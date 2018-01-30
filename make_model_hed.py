import sys, os
sys.path.insert(0, '/home/liang/caffe/python')
import caffe
from caffe import layers as L, params as P
from math import ceil
from caffe.coord_map import crop

def conv_relu(bottom, num_out, ks=3, stride=1, pad=1, lr=[1,1,2,0]):
	conv = L.Convolution(bottom, kernel_size=ks, stride=stride,num_output=num_out, pad=pad, weight_filler=dict(type='xavier'), param=[dict(lr_mult=lr[0], decay_mult=lr[1]), dict(lr_mult=lr[2], decay_mult=lr[3])])
	return conv, L.ReLU(conv, in_place=True)

#def conv_relu(bottom,num_out,kernel=3,stride=1,pad=1,weight=dict(type='xavier'),lr=[1,1,2,0]):
	#conv1=conv3x3(bottom,num_out,kernel,stride,pad,weight,lr)
	#relu1=L.ReLU(conv1,in_place=True)
	#return conv1,relu1
def conv1x1(bottom,num_out=1,kernel=1,stride=1,pad=0,weight=dict(type='constant'),lr=[0.01,1,0.02,0]):
	return L.Convolution(bottom, kernel_size=1,num_output=1, weight_filler=weight,param=[dict(lr_mult=lr[0], decay_mult=lr[1]), dict(lr_mult=lr[2], decay_mult=lr[3])])
def max_pool(bottom,kernel=2,stride=2):
	return L.Pooling(bottom,pool=P.Pooling.MAX, kernel_size=kernel, stride=stride)
def upsample(bottom, stride):
	s, k, pad = stride, 2 * stride, int(ceil(stride-1)/2)
	name = "upsample%d"%s
  	return L.Deconvolution(bottom, name=name, convolution_param=dict(num_output=1,kernel_size=k, stride=s, pad=pad, weight_filler = dict(type="bilinear")),param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])

def net(split):
	n=caffe.NetSpec()
	loss_param=dict(normalize=False)
	if split == 'train':
		data_params=dict(mean=(104.00699, 116.66877, 122.67892))
		data_params['root']='/home/liang/rcf/data/HED-BSDS'
		data_params['source']='bsds_pascal_train_pair.lst'
		data_params['shuffle']=True
		n.data,n.label=L.Python(module='ImageLabelmapData',layer='ImageLabelmapDataLayer',ntop=2,param_str=str(data_params))
	elif split == 'test':
		n.data=L.Input(name='data',input_param=dict(shape=dict(dim=[1,3,500,500])))
	else:
		raise Exception('Invalid split')
	#vgg architecture
	n.conv1_1,n.relu1_1=conv_relu(n.data,num_out=64)
	n.conv1_2,n.relu1_2=conv_relu(n.relu1_1,num_out=64)
	n.pool1=max_pool(n.relu1_2)

	n.conv2_1,n.relu2_1=conv_relu(n.pool1,num_out=128)
	n.conv2_2,n.relu2_2=conv_relu(n.relu2_1,num_out=128)
	n.pool2=max_pool(n.relu2_2)

	n.conv3_1,n.relu3_1=conv_relu(n.pool2,num_out=256)
	n.conv3_2,n.relu3_2=conv_relu(n.relu3_1,num_out=256)
	n.conv3_3,n.relu3_3=conv_relu(n.relu3_2,num_out=256)
	n.pool3=max_pool(n.relu3_3)
	
	n.conv4_1,n.relu4_1=conv_relu(n.pool3,num_out=512)
	n.conv4_2,n.relu4_2=conv_relu(n.relu4_1,num_out=512)
	n.conv4_3,n.relu4_3=conv_relu(n.relu4_2,num_out=512)
	n.pool4=max_pool(n.relu4_3)

	n.conv5_1,n.relu5_1=conv_relu(n.pool4,num_out=512,lr=[100,1,200,0])
	n.conv5_2,n.relu5_2=conv_relu(n.relu5_1,num_out=512,lr=[100,1,200,0])
	n.conv5_3,n.relu5_3=conv_relu(n.relu5_2,num_out=512,lr=[100,1,200,0])

	#conv1
	n.dsn1=conv1x1(n.conv1_2)
	n.dsn1_crop=crop(n.dsn1,n.data)

	if split=='train':
		n.dsn1_loss=L.SigmoidCrossEntropyLoss(n.dsn1_crop,n.label)
	else:
		n.sigmoid_dsn1=L.Sigmoid(n.dsn1_crop)


	#conv2
	n.dsn2=conv1x1(n.conv2_2)
	n.dsn2_up=upsample(n.dsn2,stride=2)
	n.dsn2_crop=crop(n.dsn2_up,n.data)
	if split=='train':
		n.dsn2_loss=L.SigmoidCrossEntropyLoss(n.dsn2_crop,n.label)
	else:
		n.sigmoid_dsn2=L.Sigmoid(n.dsn2_crop)

	#conv3
	n.dsn3=conv1x1(n.conv3_3)
	n.dsn3_up=upsample(n.dsn3,stride=4)
	n.dsn3_crop=crop(n.dsn3_up,n.data)
	if split=='train':
		n.dsn3_loss=L.SigmoidCrossEntropyLoss(n.dsn3_crop,n.label)
	else:
		n.sigmoid_dsn3=L.Sigmoid(n.dsn3_crop)

	#conv4
	n.dsn4=conv1x1(n.conv4_3)
	n.dsn4_up=upsample(n.dsn4,stride=8)
	n.dsn4_crop=crop(n.dsn4_up,n.data)
	if split=='train':
		n.dsn4_loss=L.SigmoidCrossEntropyLoss(n.dsn4_crop,n.label)
	else:
		n.sigmoid_dsn4=L.Sigmoid(n.dsn4_crop)



	#conv5
	n.dsn5=conv1x1(n.conv5_3)
	n.dsn5_up=upsample(n.dsn5,stride=16)
	n.dsn5_crop=crop(n.dsn5_up,n.data)
	if split=='train':
		n.dsn5_loss=L.SigmoidCrossEntropyLoss(n.dsn5_crop,n.label)
	else:
		n.sigmoid_dsn5=L.Sigmoid(n.dsn5_crop)

	
	#concat
	n.concat_5=L.Concat(n.dsn1_crop,n.dsn2_crop,n.dsn3_crop,n.dsn4_crop,n.dsn5_crop,name='concat',concat_param=dict(concat_dim=1))
	n.dsn=L.Convolution(n.concat_5,name='dsn',num_output=1,kernel_size=1,param=[dict(lr_mult=0.001, decay_mult=1), dict(lr_mult=0.002, decay_mult=0)],weight_filler=dict(type='constant', value=0.2))
	if split=='train':
		n.fuse_loss=L.SigmoidCrossEntropyLoss(n.dsn,n.label)
	else:
		n.sigmoid_fuss=L.Sigmoid(n.dsn)
	return n.to_proto()
	
	
def make_net():
	with open('hed_train_val.prototxt','w') as f:
		f.write(str(net('train')))
	with open('hed_test.prototxt','w') as f:
		f.write(str(net('test')))



if __name__ == '__main__':
	make_net()

