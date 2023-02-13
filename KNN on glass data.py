# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:10:11 2022

@author: Dell
"""

import numpy as np
import pandas as pd

df=pd.read_csv("glass.csv")
df.head()
df.shape
df.duplicated()
df[df.duplicated()]
df.drop_duplicates()
df.info()

#spliting the data

X=df.iloc[:,0:9]

X.corr()    #There is realtion btw Ri and Ca..
X.corr().to_csv("KNN.csv")

X1=df.iloc[:,1:9]

Y=df["Type"]

#Standidization

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
#X1 columns Standidization
X1["RI"]=MM.fit_transform(X1[["RI"]])

X1["Na"]=MM.fit_transform(X1[["Na"]])

X1["Mg"]=MM.fit_transform(X1[["Mg"]])

X1["Al"]=MM.fit_transform(X1[["Al"]])

X1["Si"]=MM.fit_transform(X1[["Si"]])

X1["K"]=MM.fit_transform(X1[["K"]])

X1["Ca"]=MM.fit_transform(X1[["Ca"]])

X1["Ba"]=MM.fit_transform(X1[["Ba"]])

X1["Fe"]=MM.fit_transform(X1[["Fe"]])



#model 1 fitting using X1
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X1,Y)
Y_pred=KNN.predict(X1)

from sklearn.metrics import accuracy_score
AC=accuracy_score(Y,Y_pred)

print(AC)

#Accuracy is 76%


from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()
#X columns Standidization
X["RI"]=MM.fit_transform(X[["RI"]])

X["Na"]=MM.fit_transform(X[["Na"]])

X["Mg"]=MM.fit_transform(X[["Mg"]])

X["Al"]=MM.fit_transform(X[["Al"]])

X["Si"]=MM.fit_transform(X[["Si"]])

X["K"]=MM.fit_transform(X[["K"]])

X["Ca"]=MM.fit_transform(X[["Ca"]])

X["Ba"]=MM.fit_transform(X[["Ba"]])

X["Fe"]=MM.fit_transform(X[["Fe"]])



#model 2 fitting using X
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X,Y)
Y_pred=KNN.predict(X)

from sklearn.metrics import accuracy_score
AC=accuracy_score(Y,Y_pred)

print(AC)

#Accuracy is 78%

prediction=KNN.predict(np.array([[1.51666,12.86,0,1.83,73.88,0.97,10.17,0,0]]))
