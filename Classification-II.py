# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:46:30 2019

@author: Himanshu Nirmal
Lab3 Part 2
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# function to print the scores
def score(model):
    print("Training Accuracy Score: {}".format(model.score(X_train,y_train.ravel())))
    mod_pred = model.predict(X_test)
    print("Test Accuracy Score for this model : {}".format(accuracy_score(y_test, mod_pred)))

# Question (a) -- read csv
data = pd.read_csv('./Default.csv', header=0)
#print(data.head())
#print(data.describe())

Y_data = data['default']

# Question (b) -- predictor selection
predictors = ['balance','income']
X = data[predictors].values
y = data[['default']].values


# Question (c) -- create test and train sets
X_train = X[0:8000, :]
X_test = X[8000:,:]
y_train = y[0:8000,:]
y_test = y[8000:,:]

# create models
# Question(d)
logRegression = LogisticRegression().fit(X_train, y_train.ravel())
# Question(e)
LDA = LinearDiscriminantAnalysis().fit(X_train, y_train.ravel())

# Question(f)
KNN1 = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train.ravel())
KNN5 = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train.ravel())
KNN10 = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train.ravel())
KNN100 = KNeighborsClassifier(n_neighbors=100).fit(X_train, y_train.ravel())

# Question(g)
QDA = QuadraticDiscriminantAnalysis().fit(X_train, y_train.ravel())

#print the scores
print("")
print("Logistic Regression: ")
score(logRegression)
print("")

print("LDA: ")
score(LDA)
print("")

print("KNN where K=1: ")
score(KNN1)
print("")

print("KNN where K=5: ")
score(KNN5)
print("")

print("KNN where K=10: ")
score(KNN10)
print("")

print("KNN where K=100: ")
score(KNN100)
print("")

print("QDA: ")
score(QDA)
