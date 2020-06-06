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
	return (x >= 0).astype(x.dtype)

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

def zero_pad(x,pad):
	return np.pad(x, (((0,0),pad,pad,(0,0))), mode='constant', constant_values = (0,0))

def de_zero_pad(x,pad):
	pH , pW = pad
	return x[:,pH:-pH,pW:-pW,:]

def conv2D(X,W,B):
	W = np.reshape(W,(1,*W.shape))
	Z = X*W + B
	Z = np.sum(Z,axis=tuple(range(1,Z.ndim)))
	return Z

class Conv2D:

	def __init__(self,kCount,kShape,pad=(0,0),stride=(1,1),activation=RELU):
		self.kShape = kShape
		self.kCount = kCount
		self.kHeight, self.kWidth, self.kChannel = kShape

		self.padHeight, self.padWidth = pad
		self.strideHeight , self.strideWidth = stride

		self.g  = activation['g']
		self.dg = activation['dg']

		self.W = np.random.randn(self.kHeight,self.kWidth,self.kChannel,self.kCount)*0.01
		self.B = np.zeros((self.kCount))

	def forward(self,Ain):
		m,inHeight,inWidth,inChannel = Ain.shape

		assert(inChannel == self.kChannel)
		assert(inHeight  >  self.kHeight)
		assert(inWidth   >  self.kWidth)

		outHeight = (inHeight-self.kHeight+2*self.padHeight)//self.strideHeight + 1
		outWidth  = (inWidth -self.kWidth +2*self.padWidth)//self.strideWidth  + 1
		outChannel = self.kCount

		self.outHeight  = outHeight
		self.outWidth   = outWidth
		self.outChannel = outChannel

		Ain_pad = zero_pad(Ain,(self.padHeight,self.padWidth))
		self.Ain_pad = Ain_pad
		self.Z = np.zeros((m,outWidth,outHeight,outChannel))

		for h in range(outHeight):
			hStart = h*self.strideHeight
			hEnd   = hStart + self.kHeight
			for w in range(outWidth):
				wStart = w*self.strideWidth
				wEnd   = wStart + self.kWidth
				for c in range(outChannel):
					self.Z[:,h,w,c] = conv2D(Ain_pad[:,hStart:hEnd,wStart:wEnd,:],
						                     self.W[:,:,:,c],
						                     self.B[c])

		self.A = self.g(self.Z)


	def backwards(self,dA):
		self.dZ = dA * self.dg(self.Z)

		dAin_pad = np.zeros(self.Ain_pad.shape)
		dW       = np.zeros(self.W.shape)
		dB       = np.zeros(self.B.shape)

		outHeight  = self.outHeight 
		outWidth   = self.outWidth  
		outChannel = self.outChannel

		for h in range(outHeight):
			hStart = h*self.strideHeight
			hEnd   = hStart + self.kHeight
			for w in range(outWidth):
				wStart = w*self.strideWidth
				wEnd   = wStart + self.kWidth
				for c in range(outChannel):
					dAin_pad[:,hStart:hEnd,wStart:wEnd,:] += self.W[:,:,:,c] * self.dZ[:,h,w,c]
					dW[:,:,:,c] += np.sum(self.Ain_pad[:,hStart:hEnd,wStart:wEnd,:] * self.dZ[:,h,w,c],axis=0)
					dB[c]       += np.sum(self.dZ[:,h,w,c],axis=0)

		self.dAin = de_zero_pad(dAin_pad,(self.padHeight,self.padWidth))


	def update(self,rate):
		self.W -= rate*self.dW
		self.B -= rate*self.dB

def pool(X,poolType):
	if poolType == 'MAX':
		return np.max(X,axis=tuple(range(1,X.ndim)))
	elif poolType == 'AVG':
		return np.average(X,axis=tuple(range(1,X.ndim)))
	else:
		raise Exception('Unknown poolType:'+str(poolType))

class Pool:

	def __init__(self,kShape,pad=(0,0),stride=(1,1),poolType='MAX'):
		self.kShape = kShape
		self.kHeight, self.kWidth = kShape
		self.poolType = poolType

		self.padHeight, self.padWidth = pad
		self.strideHeight , self.strideWidth = stride


	def forward(self,Ain):
		self.Ain = Ain

		m,inHeight,inWidth,inChannel = Ain.shape

		assert(inHeight  >  self.kHeight)
		assert(inWidth   >  self.kWidth)

		outHeight = (inHeight-self.kHeight+2*self.padHeight)//self.strideHeight + 1
		outWidth  = (inWidth -self.kWidth +2*self.padWidth)//self.strideWidth  + 1
		outChannel = inChannel

		self.outHeight  = outHeight
		self.outWidth   = outWidth
		self.outChannel = outChannel

		assert((inHeight-self.kHeight+2*self.padHeight)%self.strideHeight == 0)
		assert((inWidth -self.kWidth +2*self.padWidth)%self.strideWidth  == 0)

		Ain_pad = zero_pad(Ain,(self.padHeight,self.padWidth))
		self.Ain_pad = Ain_pad
		self.A = np.zeros((m,outWidth,outHeight,outChannel))

		for h in range(outHeight):
			hStart = h*self.strideHeight
			hEnd   = hStart + self.kHeight
			for w in range(outWidth):
				wStart = w*self.strideWidth
				wEnd   = wStart + self.kWidth
				for c in range(outChannel):
					self.A[:,h,w,c] = pool(Ain_pad[:,hStart:hEnd,wStart:wEnd,c],self.poolType)


	def backwards(self,dA):
		dAin_pad = np.zeros(self.Ain_pad.shape)

		outHeight  = self.outHeight 
		outWidth   = self.outWidth  
		outChannel = self.outChannel

		for h in range(outHeight):
			hStart = h*self.strideHeight
			hEnd   = hStart + self.kHeight
			for w in range(outWidth):
				wStart = w*self.strideWidth
				wEnd   = wStart + self.kWidth
				for c in range(outChannel):
					if self.poolType == 'MAX':
						#print(self.Ain_pad[:,hStart:hEnd,wStart:wEnd,c].shape)
						#print(np.reshape(self.A[:,h,w,c],[*self.A[:,h,w,c].shape,1,1]).shape)
						#print(dAin_pad[:,hStart:hEnd,wStart:wEnd,c].shape)
						
						mask = self.Ain_pad[:,hStart:hEnd,wStart:wEnd,c] == np.reshape(self.A[:,h,w,c],[*self.A[:,h,w,c].shape,1,1])
					elif self.poolType == 'AVG':
						mask = 1/(self.kWidth*self.kHeight)
					dAin_pad[:,hStart:hEnd,wStart:wEnd,c] += dA[:,hStart:hEnd,wStart:wEnd,c] * mask

		self.dAin = de_zero_pad(dAin_pad,(self.padHeight,self.padWidth))

	def update(self,rate):
		pass

class Flatten():
	def __init__(self):
		pass

	def forward(self,Ain):
		self.Ain = Ain
		m = self.Ain.shape[0]
		self.A = Ain.reshape([m,-1]).T # FC are in the shape [dim,m]

	def backwards(self,dA):
		self.dAin = np.reshape(dA.T,self.Ain.shape)

	def update(self,rate):
		pass

###### Test ########

m = 100
img_H = img_W = 32
img_C = 3

EPOCHS = 1
LEARNING_RATE = 1e-1

x = np.random.rand(m,img_H,img_W,img_C)
y = np.random.randint(2, size=m).reshape(1,m)

kc1 = 4
kc2 = 4

l3_dim = (16,144)
l4_dim = (1,l3_dim[0])

c1 = Conv2D(kc1,(3,3,img_C),(0,0),(1,1),RELU)
m1 = Pool((2,2),(0,0),(2,2),'MAX')
c2 = Conv2D(kc2,(4,4,kc1),(0,0),(1,1),RELU)
m2 = Pool((2,2),(0,0),(2,2),'MAX')

f3 = Flatten()

l3 = FullConnected(l3_dim,RELU)
l4 = FullConnected(l4_dim,SIGMOID)

for e in range(EPOCHS):

	c1.forward(x)
	m1.forward(c1.A)

	c2.forward(m1.A)
	m2.forward(c2.A)

	f3.forward(m2.A)

	l3.forward(f3.A)
	l4.forward(l3.A)

	loss = nll(l4.A,y)
	dY   = d_nll_y(l4.A,y)

	l4.backwards(dY)
	l3.backwards(l4.dAin)

	f3.backwards(l3.dAin)

	m2.backwards(f3.dAin)
	c2.backwards(m2.dAin)

	m1.backwards(c2.dAin)
	c1.backwards(c1.dAin)

	c1.update(LEARNING_RATE)
	c2.update(LEARNING_RATE)
	l3.update(LEARNING_RATE)
	l4.update(LEARNING_RATE)

	exit()
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
