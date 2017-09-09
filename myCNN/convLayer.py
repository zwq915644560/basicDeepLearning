#!/usr/bin/python
# -*- coding:utf-8 -*-

'a simple convolutional layer.'

__author__='zhangwenqiang'

import numpy as np 
from utils import *

class ConvLayer(object):
	def __init__(self, input_width, input_height, channel_number,
		         filter_width, filter_height, filter_number,
		         activator, zero_padding, stride, learning_rate):
		self.input_width=input_width
		self.input_height=input_height
		self.channel_number=channel_number
		self.filter_width=filter_width
		self.filter_height=filter_height
		self.filter_number=filter_number
		self.activator=activator
		self.zero_padding=zero_padding
		self.stride=stride
		self.learning_rate=learning_rate

		self.output_width=ConvLayer.calculate_output_size(
			self.input_width, self.filter_width,
			self.zero_padding, self.stride)
		self.output_height=ConvLayer.calculate_output_size(
			self.input_height, self.filter_height,
			self.zero_padding, self.stride)
		self.output_array=np.zeros((self.filter_number,
			self.output_height, self.output_width))
		self.filters=[]
		for i in range(self.filter_number):
			self.filters.append(Filter(filter_width, filter_height, channel_number))

	@staticmethod
	def calculate_output_size(input_size, filter_size,
		zero_padding, stride):
		return (input_size+2*zero_padding-filter_size)/stride+1

	def forward(self, input_array):
		self.input_array=input_array
		self.padded_input_array=padding(input_array, self.zero_padding)
		for i in range(self.filter_number):
			filter=self.filters[i]
			conv(self.padded_input_array,
				filter.get_weights(), filter.get_bias(),
				self.stride, self.output_array[i])
		element_wise_op(self.output_array,
			self.activator.forward)


	'''
	卷积层反向传播算法的实现, 反向传播算法需要完成几个任务：
	1. 将误差项传递到上一层。
	2. 计算每个参数的梯度。
	3. 更新参数。
	'''
	def backward(self, sensitivity_array, activator):
		self.bp_sensitivity_map(sensitivity_array, activator)
		self.bp_gradient(sensitivity_array)

	def bp_sensitivity_map(self, sensitivity_array, activator):
		'''
		计算传递到上一层的sensitivity map
		sensitivity_array: 本层的sensitivity map
		activator: 上一层的激活函数
		'''
		# 处理卷积步长，对原始sensitivity map进行扩展
		expanded_array = self.expand_sensitivity_map(
				sensitivity_array)
		# full卷积，对sensitivitiy map进行zero padding
		# 虽然原始输入的zero padding单元也会获得残差
		# 但这个残差不需要继续向上传递，因此就不计算了
		expanded_width = expanded_array.shape[2]
		zp = (self.input_width +  
			self.filter_width - 1 - expanded_width) / 2
		padded_array = padding(expanded_array, zp)
		# 初始化delta_array，用于保存传递到上一层的
		# sensitivity map
		self.delta_array = np.zeros((self.channel_number,
			self.input_height, self.input_width))
		# 对于具有多个filter的卷积层来说，最终传递到上一层的
		# sensitivity map相当于所有的filter的
		# sensitivity map之和
		for i in range(self.filter_number):
			filter=self.filters[i]
        	# 将filter权重翻转180度
			flipped_weights = np.array(map(
        		lambda i: np.rot90(i, 2), 
        		filter.get_weights()))
        	# 计算与一个filter对应的delta_array

      		delta_array=np.zeros((self.channel_number,
     			self.input_height, self.input_width))

      		for d in range(delta_array.shape[0]):
      			conv(padded_array[i], flipped_weights[d], 0, 1, delta_array[d])
			self.delta_array+=delta_array
		# 将计算结果与激活函数的偏导数做element-wise乘法操作
		derivative_array = np.array(self.input_array)
		element_wise_op(derivative_array, 
                        activator.backward)
		self.delta_array *= derivative_array

	def bp_gradient(self, sensitivity_array):
		# 处理卷积步长，对原始sensitivity map进行扩展
		expanded_array = self.expand_sensitivity_map(
            sensitivity_array)
		for f in range(self.filter_number):
            # 计算每个权重的梯度
			filter = self.filters[f]
			for d in range(filter.weights.shape[0]):
				conv(self.padded_input_array[d], 
					expanded_array[f],
					0, 1, filter.weights_grad[d])
            # 计算偏置项的梯度
			filter.bias_grad = expanded_array[f].sum()
	
	def update(self):
		'''
        按照梯度下降，更新权重
		'''
		for filter in self.filters:
			filter.update(self.learning_rate)

	def expand_sensitivity_map(self, sensitivity_array):
		depth = sensitivity_array.shape[0]
		# 确定扩展后sensitivity map的大小
		# 计算stride为1时sensitivity map的大小
		expanded_width = (self.input_width - 
			self.filter_width + 2 * self.zero_padding + 1)
		expanded_height = (self.input_height - 
			self.filter_height + 2 * self.zero_padding + 1)
		# 构建新的sensitivity_map
		expand_array = np.zeros((depth, expanded_height, 
								 expanded_width))
		# 从原始sensitivity map拷贝误差值
		for i in range(self.output_height):
			for j in range(self.output_width):
				i_pos = i * self.stride
				j_pos = j * self.stride
				expand_array[:,i_pos,j_pos] = sensitivity_array[:,i,j]
		return expand_array



	

# Filter类保存了卷积层的参数以及梯度，并且实现了用梯度下降算法来更新参数
class Filter(object):
	def __init__(self, width, height, depth):
		self.weights=np.random.uniform(-1e-4, 1e-4,
			(depth, height, width))
		
		self.bias=0
		self.weights_grad=np.zeros(self.weights.shape)
		self.bias_grad=0

	def __repr__(self):
		return 'filter weights:\n%s\nbias:\n%s' %(
			repr(self.weights), repr(self.bias))

	def get_weights(self):
		return self.weights

	def get_bias(self):
		return self.bias

	def update(self, learning_rate):
		self.weights-=self.weights_grad*learning_rate
		self.bias-=self.bias_grad*learning_rate

class ReluActivator(object):
	def forward(self, weighted_input):
		return max(0, weighted_input)

	def backward(self, output):
		return 1 if output>0 else 0

class IdentityActivator(object):
	def forward(self, weighted_input):
		return weighted_input

	def backward(self, output):
		return 1 

# 测试卷积层前向传播的正确性
if __name__=='__main__':
	conv1=ConvLayer(5, 5, 3, 
					3, 3, 2, 
					ReluActivator(), 1, 2, 0.01)
	input=np.fromfunction(lambda z,y,x:z+y+z, (3,5,5))
	conv1.forward(input)
	print conv1.output_array

