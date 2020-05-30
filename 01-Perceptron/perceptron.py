import numpy as np 
import matplotlib.pyplot as plt 

class perceptron():
	
	def __init__(self,shape,mu,dev,epsilon):
		self.weights = dev*np.random.randn(shape) + mu
		self.epsilon = epsilon
	
	def train(self,data,epochs):
		for _ in range(epochs):
			for x , y in data:
				if self.weights.dot(x) > 0 and y == 0:
					self.weights = self.weights - self.epsilon*x
				elif self.weights.dot(x) <= 0 and y == 1:
					self.weights = self.weights + self.epsilon*x
				else:
					pass
				#print('Epoch:',epochs,'Data point:',i,'Weights',self.weights,'Product',self.weights.dot(x))

	def test(self,data):
		acc = 0
		for x , y in data:
			y_ = self.predict(x)
			if y_ == y:
				acc += 1
		return acc/len(data)

	def predict(self,x):
		y = self.weights.dot(x)
		if y > 0:
			y = 1
		else:
			y = 0
		return y

##-------------------Or-Example--------------------------##
'''
Or Data
'''
data = [(np.array([0,0,1]),0),
		(np.array([0,1,1]),1),
		(np.array([1,0,1]),1),
		(np.array([1,1,1]),1)]

'''
Training and Testing my_model
'''
my_model = perceptron(shape=3,mu=0,dev=1,epsilon=0.1)
my_model.train(data=data,epochs=100)
print('Acc:',my_model.test(data=data))

'''
Plot Points
'''
plt.plot([0],[0],'bo')
plt.plot([0,1,1],[1,0,1],'ro')
'''
Plot Line
'''
line_plot_points = np.linspace(-1,2)
output_to_plot_points = - (my_model.weights[2] + my_model.weights[0]*line_plot_points) / my_model.weights[1]
plt.plot(line_plot_points,output_to_plot_points,'g')

'''
Plot Window Call
'''
plt.axis([-1,2,-1,2])
plt.show()

##-------------------------------------------------------##


##--------------------Xor-Example------------------------##
'''
Xor Data
'''
data = [(np.array([0,0,1]),0),
		(np.array([0,1,1]),1),
		(np.array([1,0,1]),1),
		(np.array([1,1,1]),0)]
'''
Transform Data

y1 = exp(−1/2||x − a||^2)
y2 = exp(−1/2||x − b||^2)

where

a = [0,0]
b = [1,1]

'''

def transform(data):
	transformed_data = []
	a = np.array([0,0,1])
	b = np.array([1,1,1])
	for x , y in data:
		y1 = np.exp(-1/2*(x-a).dot(x-a))
		y2 = np.exp(-1/2*(x-b).dot(x-b))
		transformed_data += [(np.array([y1,y2,1]),y)]
	return transformed_data

transformed_data = transform(data)

'''
Training and Testing my_model
'''
my_model = perceptron(shape=3,mu=0,dev=1,epsilon=1)
my_model.train(data=transformed_data,epochs=100)
print('Acc:',my_model.test(data=transformed_data))

'''
Plot Points
'''
plt.plot([transformed_data[0][0][0],transformed_data[3][0][0]],[transformed_data[0][0][1],transformed_data[3][0][1]],'bo')
plt.plot([transformed_data[1][0][0],transformed_data[2][0][0]],[transformed_data[1][0][1],transformed_data[2][0][1]],'ro')
'''
Plot Line
'''
line_plot_points = np.linspace(-1,2)
output_to_plot_points = - (my_model.weights[2] + my_model.weights[0]*line_plot_points) / my_model.weights[1]
plt.plot(line_plot_points,output_to_plot_points,'g')

'''
Plot Window Call
'''
plt.axis([-1,2,-1,2])
plt.show()

##-------------------------------------------------------##