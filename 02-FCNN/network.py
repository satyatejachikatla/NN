import numpy as np
from functools import reduce

def sigmoid(x):
	return 1/(1+np.exp(-x))
def d_sigmoid(x):
	s = sigmoid(x)
	return s*(1-s)

def relu(x):
	return np.maximum(0,x)
def d_relu(x):
	return (x > 0).astype(x.dtype)

SIGMOID = {'g' : sigmoid , 'dg' : d_sigmoid}
RELU    = {'g' : relu    , 'dg' : d_relu}

def nll(y_,y):
	epsilon = 1e-300
	m = y.shape[1]
	return -np.sum(y*np.log(y_+epsilon)+(1-y)*np.log(1-y_+epsilon))/m

def d_nll_y(y_,y):
	epsilon = 1e-300
	return -y/(y_+epsilon) + (1-y)/(1-y_+epsilon)

class FullConnected:
	def __init__(self,shape,activation):
		outSize,inSize = shape
		self.W = np.random.randn(outSize,inSize)*0.01
		self.B = np.zeros((outSize,1))
		self.g = activation['g']
		self.dg = activation['dg']

	def forward(self, Ain):
		self.Ain = Ain
		self.Z  = np.dot(self.W,self.Ain) + self.B
		self.A  = self.g(self.Z)

	def backwards(self,dA):
		m = dA.shape[1]
		self.dZ = dA * self.dg(self.Z)
		self.dW = np.dot(self.dZ, self.Ain.T)/m
		self.dB = np.sum(self.dZ, axis=1,keepdims=True)/m

		self.dAin = np.dot(self.W.T,self.dZ)

	def update(self,rate):
		self.W -= rate*self.dW
		self.B -= rate*self.dB


###### Test ########

EPOCHS = 10000
LEARNING_RATE = 1e-1

m = 100
split = 10
x_dim = 5
x = np.random.randn(x_dim,m)
y = np.random.randint(2, size=m).reshape(1,m)

l1_dim = (5,x_dim)
l2_dim = (10,l1_dim[0])
l3_dim = (1,l2_dim[0])

l1 = FullConnected(l1_dim,RELU)
l2 = FullConnected(l2_dim,RELU)
l3 = FullConnected(l3_dim,SIGMOID)

for e in range(EPOCHS):
	loss = 0
	for i in range(m//split):
		x_batch = x[:,i*split:(i+1)*split]
		y_batch = y[:,i*split:(i+1)*split]
		l1.forward(x_batch)
		l2.forward(l1.A)
		l3.forward(l2.A)

		loss += nll(l3.A,y_batch)
		dY   = d_nll_y(l3.A,y_batch)

		l3.backwards(dY)
		l2.backwards(l3.dAin)
		l1.backwards(l2.dAin)

		l1.update(LEARNING_RATE)
		l2.update(LEARNING_RATE)
		l3.update(LEARNING_RATE)

	if e%100 == 0:
		print(loss)

l1.forward(x)
l2.forward(l1.A)
l3.forward(l2.A)
loss = nll(l3.A,y)

print('Final Loss:',loss)
print('Actual:',y)
print('Predicted:',np.round(l3.A).astype(int))

acc = 1 - np.sum(np.abs(y-np.round(l3.A)))/m
print('Acc:',acc)

