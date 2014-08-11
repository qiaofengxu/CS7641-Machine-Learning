import pylab as pl
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# Load the diabetes dataset
dataset = datasets.load_diabetes()

print dataset.data[0,:]  # should print first row, all columns

print dataset.data   # print feature names?

print "---------------------"

model = LinearRegression()

model.fit(dataset.data, dataset.target)

print "model:\n", model

# Split the targets into training/testing sets
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test = diabetes.target[-20:]

# make predictions

expected = dataset.target

predicted = model.predict(dataset.data)

# summarize the fit of the model

mse = np.mean((predicted-expected)**2)

print mse

print model.score(dataset.data, dataset.target)

print model.coef_

