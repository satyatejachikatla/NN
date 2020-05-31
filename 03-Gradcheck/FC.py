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


###### Test ########

m = 100
x_dim = 5
x = np.random.randn(x_dim,m)
y = np.random.randint(2, size=m).reshape(1,m)

l1_dim = (5,x_dim)
l2_dim = (10,l1_dim[0])
l3_dim = (1,l2_dim[0])

l1 = FullConnected(l1_dim,RELU)
l2 = FullConnected(l2_dim,RELU)
l3 = FullConnected(l3_dim,SIGMOID)

def full_pass():
	global l1,l2,l3

	l1.forward(x)
	l2.forward(l1.A)
	l3.forward(l2.A)

	loss = nll(l3.A,y)
	dY   = d_nll_y(l3.A,y)

	l3.backwards(dY)
	l2.backwards(l3.dAin)
	l1.backwards(l2.dAin)

def forward_pass():
	global l1,l2,l3

	l1.forward(x)
	l2.forward(l1.A)
	l3.forward(l2.A)

	loss = nll(l3.A,y)

	return loss

def load_parameters(WB_per_dim):
	global l1,l2,l3

	l1.W = WB_per_dim[0]['W']
	l1.B = WB_per_dim[0]['B']

	l2.W = WB_per_dim[1]['W']
	l2.B = WB_per_dim[1]['B']

	l3.W = WB_per_dim[2]['W']
	l3.B = WB_per_dim[2]['B']

# dtheta
full_pass()
theta = np.concatenate([l1.W.flatten(),l2.W.flatten(),l3.W.flatten(),l1.B.flatten(),l2.B.flatten(),l3.B.flatten()])
dtheta = np.concatenate([l1.dW.flatten(),l2.dW.flatten(),l3.dW.flatten(),l1.dB.flatten(),l2.dB.flatten(),l3.dB.flatten()])

def ThetaToDims(dims,theta):
	dw_offsets = [0] + [np.product(dim) for dim in dims]
	for i in range(1,len(dw_offsets)):
		dw_offsets[i] = dw_offsets[i]+dw_offsets[i-1]

	db_offsets = [dw_offsets[-1]] + [dim[0] for dim in dims]
	for i in range(1,len(db_offsets)):
		db_offsets[i] = db_offsets[i]+db_offsets[i-1]

	WB_per_dim = []
	for i in range(len(dims)):
		outSize,inSize = dims[i]
		W = theta[dw_offsets[i]:dw_offsets[i+1]].reshape(outSize,inSize)
		B = theta[db_offsets[i]:db_offsets[i+1]].reshape(outSize,1)
		WB_per_dim.append({'W':W,'B':B})

	return WB_per_dim

dims = [l1_dim,l2_dim,l3_dim]
dtheta_approx = np.zeros(dtheta.shape)

epsilon = 1e-10

for i in range(len(theta)):
	ltheta_p = np.copy(theta)
	ltheta_p[i] += epsilon

	WB_per_dim = ThetaToDims(dims,ltheta_p)
	load_parameters(WB_per_dim)

	loss_p = forward_pass()

	# ------------------------------------- #

	ltheta_n = np.copy(theta)
	ltheta_n[i] -= epsilon

	WB_per_dim = ThetaToDims(dims,ltheta_n)
	load_parameters(WB_per_dim)

	loss_n = forward_pass()

	dtheta_approx[i] = (loss_p-loss_n)/(2*epsilon)

#### dtheta to dtheta_approx #####
numerator = np.linalg.norm(dtheta-dtheta_approx)
denominator = np.linalg.norm(dtheta) + np.linalg.norm(dtheta_approx)

error = numerator/denominator

print('Error:',error)
print('Epsilon:',epsilon)

if(error > 2*1e-3):
	print('Error in gradients')
	error_terms = np.abs(dtheta-dtheta_approx)
	error_terms = ThetaToDims(dims,error_terms)

	dtheta_approx = ThetaToDims(dims,dtheta_approx)
	dtheta = ThetaToDims(dims,dtheta)

	print('l2.dB : ',dtheta[1]['B'])
	print('l2.dB_approx : ',dtheta_approx[1]['B'])
	print('l2.B_diff : ',error_terms[1]['B'])

	print('numerator:',numerator)
	print('denominator:',denominator)

else:
	print('Ok')