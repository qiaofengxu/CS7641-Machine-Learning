#DecisionTreeClassifier
from sklearn import datasets 
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier 

# load the iris datasets 

dataset=datasets.load_iris() 

print(dataset.data.shape)
print(dataset.feature_names)
print(dataset.target_names)

print(dataset.keys())
print(dataset)

# fit a CART model to the data 

model=DecisionTreeClassifier() 
model.fit(dataset.data,dataset.target) 
print(model)

# make predictions

expected=dataset.target 
predicted=model.predict(dataset.data)

# summarize the fit of the model 

print(metrics.classification_report(expected,predicted)) 
print(metrics.confusion_matrix(expected,predicted))
print(model.feature_importances_)