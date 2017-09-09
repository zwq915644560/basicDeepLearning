#!/usr/bin/python
# -*- coding:utf-8 -*-

'some frequently used functions.'

__author__='zhangwenqiang'

import numpy as np

# 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array,
                       op_flags=['readwrite']):
        i[...] = op(i)

# conv函数实现了2维和3维数组的卷积
def conv(input_array, kernel_array, bias, stride, output_array):
	output_width=output_array.shape[1]
	output_height=output_array.shape[0]
	kernel_width=kernel_array.shape[-1]
	kernel_height=kernel_array.shape[-2]
	for i in range(output_height):
		for j in range(output_width):
			output_array[i][j]=(get_patch(input_array, i, j,
				kernel_width, kernel_height, stride)*kernel_array
			).sum()+bias

# 根据filter的窗口大小从输入数组中获取相应的卷积窗口
def get_patch(input_array, i, j,
	window_width, window_height, stride):
	#span1=int(window_height/2)
	#span2=int(window_width/2)
	if input_array.ndim==2:
		return input_array[i*stride:i*stride+window_height,
						   j*stride:j*stride+window_width]
	elif input_array.ndim==3:
		return input_array[:, i*stride:i*stride+window_height,
						   j*stride:j*stride+window_width]

# 为数组增加Zero padding
def padding(input_array, zp):
	if zp==0:
		return input_array
	else:
		if input_array.ndim==2:
			input_height=input_array.shape[0]
			input_width=input_array.shape[1]
			padded_array=np.zeros((input_height+2*zp,
							 	   input_width+2*zp))
			padded_array[zp:zp+input_height,
						 zp:zp+input_width]=input_array
			return padded_array
		elif input_array.ndim==3:
			input_depth=input_array.shape[0]
			input_height=input_array.shape[1]
			input_width=input_array.shape[2]
			padded_array=np.zeros((input_depth,
								   input_height+2*zp,
								   input_width+2*zp))
			padded_array[:,
			             zp:zp+input_height,
			             zp:zp+input_width]=input_array
			return padded_array
