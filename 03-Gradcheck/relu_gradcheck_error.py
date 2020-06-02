import numpy as np

def relu(x):
	return np.maximum(0,x)
def d_relu(x):
	return (x >= 0).astype(x.dtype)

def relu_gradcheck(x):
	

	J_p = relu(x+epsilon)
	J_n = relu(x-epsilon)

	dJ_approx = (J_p-J_n)/(2*epsilon)
	dJ = d_relu(x)

	numerator = np.linalg.norm(dJ-dJ_approx)
	denominator = np.linalg.norm(dJ)+np.linalg.norm(dJ_approx)

	error = numerator/denominator

	return dJ,dJ_approx,error

epsilon = 1e-7
steps = 100

for eps in np.linspace(-epsilon,epsilon,steps):
	x = np.array([eps])
	dJ,dJ_approx,error = relu_gradcheck(x)
	print('---------------------------------')
	print('X:',x)

	print('dJ:',dJ)
	print('dJ_approx:',dJ_approx)
	
	print('Error:',error)

print('---------------------------------')
print('For the point x = 0 derivative in neighbourhood of\n' 
	  '(-epsilon,epsilon) fails to give a accuate value as the\n'
	  'relu is not continuous. Hence gradient checking cannot\n'
	  'applied for relu.')