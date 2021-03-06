# -*- coding:utf-8 -*-

'''
定义一个感知器
其构成包含输入数据的个数，权重集合，偏置项和激活函数
需要对该感知器进行传入训练数据，从而不断调整权重参数和偏置。
'''
class Perceptron(object):
	def __init__(self, input_num, activator):
		'''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
		self.activator=activator
		 # 权重向量初始化为0
		self.weights=[0.0 for _ in range(input_num)]
		 # 权重向量初始化为0
		self.bias=0.0

	def __str__(self):
		'''
		打印学习到的权重、偏置项
		'''
		return "weights\t:%s\nbias\t:%f\n" %(self.weights, self.bias)

	def predict(self, input_vec):
		'''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
		temp=zip(input_vec, self.weights)
		net=reduce(lambda a, b: a+b, map(lambda (x, w): w*x, temp))+self.bias
		out=self.activator(net)
		return out

	def train(self, input_vecs, labels, iteration, rate):
		'''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
		for i in range(iteration):
			self._one_iteration(input_vecs, labels, rate)

	def _one_iteration(self, input_vecs, labels, rate):
		'''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
		samples=zip(input_vecs,labels)
		# 对每个样本，按照感知器规则更新权重
		for (input_vec, label) in samples:
			# 计算感知器在当前权重下的输出
			out=self.predict(input_vec)
			# 更新权重
			# 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        	# 变成[(x1,w1),(x2,w2),(x3,w3),...]
			delta=label-out
			self.weights=map(lambda (w, x): w+rate*delta*x, zip(self.weights, input_vec))
			# 更新bias
			self.bias=self.bias+rate*delta

'''
利用感知器实现and函数
'''
def f(x):
	'''
	定义一个激活函数
	'''
	return 1 if x>0 else 0

def get_training_dataset():
    '''
    基于and真值表构建训练数据
    '''
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels    

def train_and_perceptron():
    '''
    使用and真值表训练感知器
    '''
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    #返回训练好的感知器
    return p

if __name__=="__main__":
	# 训练and感知器
	and_perceptron=train_and_perceptron()
	# 打印训练获得的权重和偏置
	print and_perceptron
	# 测试
	print "1 and 1 = %d" % and_perceptron.predict([1, 1])
	print "1 and 0 = %d" % and_perceptron.predict([1, 0])
	print "0 and 1 = %d" % and_perceptron.predict([0, 1])
	print "0 and 0 = %d" % and_perceptron.predict([0, 0])
