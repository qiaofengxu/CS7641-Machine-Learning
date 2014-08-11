#CrossValidationClassification
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
#loadtheirisdatasets
dataset=datasets.load_iris()
#preparecrossvalidationfolds
num_folds=10
num_instances=len(dataset.data) 
kfold=cross_validation.KFold(n=num_instances,n_folds=num_folds) 
#prepareaLogisticRegressionmodel
model=LogisticRegression()
#evaluatethemodel k-foldcrossvalidation 
results=cross_validation.cross_val_score(model,dataset.data,dataset.target,cv=kfold) 
#displaythemeanclassificationaccuracyoneachfold
print(results)
#displaythemeanandstdevoftheclassificationaccuracy
print(results.mean())
print(results.std())