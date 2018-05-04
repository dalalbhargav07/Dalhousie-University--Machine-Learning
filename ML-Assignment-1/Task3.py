# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:59:17 2018

@author: dalalbhargav07
"""

#Importing the numpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
wtrain = np.loadtxt('wine.train', delimiter = ',')
wtrain_X = np.delete(wtrain,0,axis=1)
wtrain_y = wtrain[:,0]

wtest = np.loadtxt('wine.test',delimiter=',')
wtest_X =  np.delete(wtest,0,axis=1)
wtest_y = wtest[:,0]

#Support Vector Machine Object
from sklearn.svm import SVC
svc = SVC(kernel='linear')
from sklearn.cross_validation import cross_val_score
svc_mean = cross_val_score(svc, wtrain_X, wtrain_y, cv = 10, scoring = 'accuracy')
svc_mn = svc_mean.mean()
print('Accuracy obtained using SVM: %.3f'%(svc_mn*100) +'%')
print('-----------------------------------------------------------------------')
print('')

#Random Forest Classifier Object
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=30)
rfc_mean = cross_val_score(rfc, wtrain_X, wtrain_y, cv = 10, scoring = 'accuracy')
rfc_mn = rfc_mean.mean()
print('Accuracy obtained using RFC: %.3f'%(rfc_mn*100) +'%')
print('-----------------------------------------------------------------------')
print('')


#Multi Layer Perceptron

import tensorflow as tf
from sklearn.model_selection import train_test_split

X=wtrain_X
y=wtrain_y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


y_train = y_train.astype(int)


feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
MLP = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=4)

MLP.fit(X_train, y_train, steps=2000, batch_size=84)
pred = list(MLP.predict(X_test))
mlp_mn = accuracy_score(y_test,pred)
print ('')
print('Accuracy obtained using MLP: %.3f'%(mlp_mn*100) +'%')
print('-----------------------------------------------------------------------')
print('')


print ('Accuracies for Various Classification Model are:')
print ('Support Vector Machine:', svc_mn*100)
print ('Random Forest:', rfc_mn*100)
print ('Multi Layer Peceptrom:', mlp_mn*100)
print('-----------------------------------------------------------------------')
print('')


x=['SVM','RFC','MLP']
y=[svc_mn*100,rfc_mn*100,mlp_mn*100]
nd = np.arange(3)
width=0.2
plt.xticks(nd-width/2., ('SVM','RFC','MLP'))
fig = plt.bar(nd, y)

plt.show()

print('It is quite clear that Random Forest classifier has the highest accuracya and hence it is chosen as the sutiable model for the given wine dataset')


rfc.fit(wtrain_X,wtrain_y)
pred = rfc.predict(wtest_X)

#To save my prediction file
#np.savetxt('pred_wine.csv',pred)

'''
Reason to choose RFC:
The first step for selecting the best model was to tune the parameter of each model. 
The approach for tunning the parameter of each was trial and error. 
Once the parameter was tuned, each model was applied on the data set using cross 
validation technique and then for each model, accuracy was calculated. 
The Random Forest Classifier (RFC) was the best performer among the varisous 
classification algortihm such as Support Vector Machine (SVM), and Multi Layer Perceptron (MLP). 
The accuracy was the measure considerd to choose the best among them. 
As can see from the plot, RFC has the highest accuracy and hence it is selected 
as best classification algorithm for wine dataset.
'''

