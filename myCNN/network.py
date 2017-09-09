#!/usr/bin/python
# -*- coding:utf-8 -*-

'implement a CNN.'

__author__='zhangwenqiang'

import numpy as np 
from convLayer import *
from poolingLayer import *
from fullConnectedLayer import *

class LeNet(object):
	def __init__(self):
		#self.params={}
		#self.params['convLayer']=1
		self.conv1=ConvLayer(28,28,1,3,3,6,ReluActivator(),1,1,0.01)
		self.pooling1=PoolingLayer(conv1.output_width,
			conv1.output_height, conv1.filter_number,
			2,2,2)
		self.conv2=ConvLayer(pooling1.output_width, pooling1.output_height, pooling1.channel_number,
			3,3,10,ReluActivator(),1,1,0.01)
		self.pooling2=PoolingLayer(conv2.output_width,
			conv2.output_height, conv2.filter_number,
			2,2,2)
		fc_input_num=pooling2.output_width*pooling2.output_height*pooling2.filter_number
		self.fc=FullConnectedLayer([fc_input_num, 100, 10])


	def predict(self, input_array):
		self.conv1.forward(input_array)
		self.pooling1.forward(self.conv1.output_array)
		self.conv2.forward(self.pooling1.output_array)
		self.pooling2.forward(self.conv2.output_array)
		return self.fc.predict(self.pooling2.output_array)

	def train(self, dataSet, labels, rate, epoch):
		