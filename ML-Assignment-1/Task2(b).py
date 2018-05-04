# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:56:41 2018

@author: dalalbhargav07
"""
#Removing warnings
import warnings
warnings.filterwarnings('ignore')
#Importing the numpy
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#Creating objecy for SVM
from sklearn.svm import SVC
svc = SVC()
from sklearn import datasets 
iris = datasets.load_iris()
#Loading the data set:
X = iris.data
y = iris.target

#Function for K-Fold:
from sklearn.cross_validation import KFold
kf = KFold(len(X),10,shuffle = True,random_state=100)
arr_mean = []

for train_index, test_index in kf:
    #print("Train INdex:", train_index,"test index:",test_index)
    
    svc.fit(X[train_index], y[train_index])
    prd = svc.predict(X[test_index])
    acc_mean = accuracy_score(y[test_index], prd) 
    arr_mean.append(acc_mean)
arr_mn = np.array(arr_mean).mean()

print ('Accuracy for Kfold script is: %.3f'%(arr_mn*100) +'%')

#Calling the cross_val_score
from sklearn.cross_validation import cross_val_score
arr_mean_cvs = cross_val_score(svc, X, y, cv = 10, scoring = 'accuracy')
arr_mn_cvs = arr_mean_cvs.mean()
print('Accuracy obtained using cross_val_score: %.3f'%(arr_mn_cvs*100) +'%')

'''
As you can see the accuracy for SVM model using cross_val_score is better than the kfold 
function which we have used. 
Hence, cross validation method can be more quick and effcient option for the 
performance of our model as the model will get more and more data for training.
'''