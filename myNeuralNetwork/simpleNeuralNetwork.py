#!/usr/bin/python
# -*- coding: utf-8 -*-

'a simple full connected neural network.'

__author__='zhangwenqiang'

import numpy as np 

class FullConnectedLayer(object):
	def __init__(self, input_size, output_size, activator):
		self.input_size=input_size
		self.output_size=output_size
		self.activator=activator

		self.W=np.random.uniform(-1, 1, (output_size, input_size))
		self.b=np.zeros((output_size, 1))

		self.output=np.zeros((output_size, 1))

	def forward(self, input_array):
		self.input=input_array
		self.output=self.activator.forward(
			np.dot(self.W, self.input)+self.b)

	def backward(self, deltaVec):  
		self.delta=self.activator.backward(self.input)*np.dot(self.W.T, deltaVec)
		self.W_grad=-1*np.dot(deltaVec, self.input.T)
		self.b_grad=-1*deltaVec

	def update(self, rate):
		self.W-=rate*self.W_grad
		self.b-=rate*self.b_grad

class SigmiodActivator(object):
	def forward(self, weighted_input):
		return 1.0/(1.0+np.exp(-1*weighted_input))
	def backward(self, output):
		return output*(1-output)

class Network(object):
	def __init__(self, layerlst):
		self.layers=[]
		for i in range(len(layerlst)-1):
			layer=FullConnectedLayer(layerlst[i], layerlst[i+1], SigmiodActivator())
			self.layers.append(layer)

	def predict(self, sample):
		output=np.array(sample).reshape(-1,1)
		for layer in self.layers:
			layer.forward(output)
			output=layer.output
		return  output

	def train(self, dataSet, labels, rate, epoch):
		for i in range(epoch):
			#for sample, label in zip(dataSet, labels):
			#	self._train_one_sample(sample, label, rate)
			for i in range(len(dataSet)):
				self._train_one_sample(dataSet[i], labels[i], rate)

	def _train_one_sample(self, sample, label, rate):
		#################
		self.predict(sample)
		self.calc_gradient(label)
		self.update_weight(rate)

	def calc_gradient(self, label):
		# output layer's delta
		out=self.layers[-1].output
		delta=self.layers[-1].activator.backward(out)*(label-out)#######  error!!!!!!
		for layer in self.layers[::-1]:
			layer.backward(delta)
			delta=layer.delta
		return delta

	def update_weight(self, rate):
		for layer in self.layers:
			layer.update(rate)








