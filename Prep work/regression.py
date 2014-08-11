import pylab as pl
import numpy as np
from scipy import linalg as LA
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation

nobs = 100

X_base = np.arange(1,nobs+1).reshape(nobs,1)

print X_base.shape

np.random.seed([4])

Y = 2.0 * X_base + 3.0 + np.random.normal(0,5,nobs).reshape(nobs,1)

print "\n---------- Coeffs with Linear Algebra -----------"

ones = np.ones((nobs,1), dtype=np.int)
X = np.hstack([ones, X_base])

XT = np.transpose(X)

coeffs = LA.inv(XT.dot(X)).dot(XT.dot(Y))

print coeffs

predicted = X.dot(coeffs)

error = predicted - Y

mse = error.T.dot(error) / nobs

print float(mse)

print "\n---------- Coeffs with scikit learn -----------"

model = LinearRegression()

model.fit(X_base, Y)

print "model:\n", model

# make predictions

expected = Y

predicted = model.predict(X_base)

# summarize the fit of the model

mse = np.mean((predicted-expected)**2)

print "MSE:", mse

print "R^2:", model.score(X_base, Y)  # Coeff of determination of the prediction

print "Coefficients:", model.coef_
print "Intercept:", model.intercept_


print "\n---------- Cross Validation -----------"

# cross validation

#prepare cross validation folds
num_folds=10
num_instances=len(X_base) 
kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds) 


#evaluate the model k-fold cross validation 

results=cross_validation.cross_val_score(model, X_base, Y, cv=kfold, scoring='mean_squared_error') 
# note that MSE sign is flipped to allow optimization.  MSE is always positive.

# display the mean accuracy on each fold (from the score method of estimator)
print "cv results:", results

# display the mean and stdev of the accuracy
print "cv mean:", results.mean()
print "cv std:", results.std()


# Plot outputs
pl.scatter(X_base, Y,  color='black')
pl.plot(X_base, model.predict(X_base), color='blue',
        linewidth=3)

pl.xticks(())
pl.yticks(())

pl.show()

