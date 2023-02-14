# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 22:23:10 2022

@author: Dell
"""

import numpy as np
import pandas as pd

df=pd.read_csv("Zoo.csv")
df.head()
df.shape

#droping the first column 
df.drop(["animal name"],axis=1,inplace=True)
df
df.shape
df.info()
#spliting
X=df.iloc[:,0:16]
X.columns
X.corr()
Y=df["type"]

#there is relation btw hair,eggs,milk.....
X1=df.drop(df.columns[[0,2,16]],axis=1)
X1.columns
Y=df["type"]


#Standidization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

X=MM.fit_transform(X)
X=pd.DataFrame(X)

X1=MM.fit_transform(X1)
X1=pd.DataFrame(X1)


#model 1 fitting using X..

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X,Y)
Y_pred=KNN.predict(X)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y,Y_pred)

#accuracy score= 95%
#There should be over fitting

################################################################
#model  2 fitting using X1..
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=5)
KNN.fit(X1,Y)
Y_pred=KNN.predict(X1)

from sklearn.metrics import accuracy_score
ac2=accuracy_score(Y,Y_pred)

#accuracy score= 94%


