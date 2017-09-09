from simpleNeuralNetwork import *
import numpy as np 

def get_training_dataset():
	trainingDataset=[[2,7,9],
					 [3,9,5],
					 [5,10,9],
					 [1,12,3],
					 [4,15,6],
					 [7,7,3]]
	#labels=np.array([[11.5], [10.4], [14.8], [9.8], [15.2], [9.8]])
	labels=[11.5, 10.4, 14.8, 9.8, 15.2, 9.8]
	return trainingDataset,labels

def train_model():
	model=Network([3,4,1])
	dataSet, labels=get_training_dataset()
	model.train(dataSet, labels, 0.1, 1000)
	return model

model=train_model()
print model.predict([6, 8, 10])

