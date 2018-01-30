#!/usr/bin/env python

from __future__ import division
import numpy as np
import sys
caffe_root = '/home/liang/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe


base_weights = '5stage-vgg.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(0)

solver = caffe.SGDSolver('solver.prototxt')

# copy base weights for fine-tuning
#solver.restore('rcf_bsds_iter_2000.solverstate')
solver.net.copy_from(base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
for i in range(3*300):
	solver.step(20)




#newsolver=caffe.SGDSolver('newsolver.prototxt')
#newsolver.net.copy_from('5stage-vgg.caffemodel')
#newsolver.step(6000)
#newsolver.net.save('newsolver_6000steps.caffemodel')
