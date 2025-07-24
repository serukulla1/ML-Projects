# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 11:55:58 2025

@author: swapna
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience\ML Projects\LogisticRegression - 2\Data\Social_Network_Ads.csv')

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=100)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=11)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred)
print(ac)


## classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

#### With standard scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

classifier.fit(X_train_sc, y_train)

y_pred_sc = classifier.predict(X_test_sc)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_sc)
print(cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred_sc)
print(ac)


#### With  scaling normalisation

from sklearn.preprocessing import Normalizer
sc_norm = Normalizer()
X_train_nm = sc_norm.fit_transform(X_train)
X_test_nm = sc_norm.transform(X_test)
classifier.fit(X_train_nm, y_train)
y_pred_nm = classifier.predict(X_test_nm)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_nm)
print(cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred_nm)
print(ac)

## classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred_nm)
print(cr)

bias=classifier.score(X_train, y_train)
variance=classifier.score(X_test, y_test)


dataset1 = pd.read_csv(r'C:\Swapna\Learning\PYTHON\FS DataScience\ML Projects\LogisticRegression - 2\Data\Future prediction1.csv')

X = dataset1.iloc[:, [2,3]].values
y = dataset1.iloc[:, -1].values
d2 = dataset1.copy()

dataset1 = dataset1.iloc[:, [2,3]].values

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
M = sc.fit_transform(dataset1)

y_pred1 = pd.DataFrame()
d2['y_pred1'] = classifier.predict(M)
d2.to_csv('final2.csv')

final2 = pd.read_csv('final2.csv')
final2




