#Perceptron
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron 
from matplotlib import pyplot as plt

print "------------- Create data set ------------"

np.random.seed([1])
X = np.random.randint(0,9, 50).reshape(25, 2)
# bias = np.ones((25,1), dtype=np.int)
# X = np.hstack([bias, X])
print X

Y = (X[:,0] + X[:, 1]).reshape(25,1)

def create_threshold(x):
	if x - 9 > 0:
		return 1
	else:
		return 0

vfunc = np.vectorize(create_threshold)

Y = np.apply_along_axis(vfunc,0,Y)

print Y

plt.scatter(X[:,0], X[:,1], marker='+', s=150, c=Y)
plt.show()


print "------------- Run Perceptron on dataset ------------"

model = Perceptron(n_iter = 100).fit(X,Y)   # number of iterations is important - 

print(model)

print "Coefficients:", model.coef_
print "Intercept:", model.intercept_

#makepredictions

for i in range(10):
	for j in range(10):
		print i, j, model.predict([i,j])


expected=Y
predicted=model.predict(X)

#summarizethe fit of the model 
mse=np.mean((predicted-expected)**2) 

print(mse) 
print(model.score(X,Y))

# look at the weights and theta
# print the coordinates, color by Y and plot the line

print "-------------SKL Perceptron - Diabetes ------------"
# load the diabetes datasets 
dataset=datasets.load_diabetes() 

# fit a Perceptron model to the data 
model=Perceptron() 
model.fit(dataset.data,dataset.target) 
print(model)

#makepredictions
expected=dataset.target 
predicted=model.predict(dataset.data)

#summarizethe fit of the model 
mse=np.mean((predicted-expected)**2) 

print(mse) 
print(model.score(dataset.data,dataset.target))