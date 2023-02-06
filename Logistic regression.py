# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 18:48:40 2022

@author: Dell
"""

import pandas as pd
import numpy as np
df=pd.read_csv("bank-full.csv",sep=';')
df
df.head()
df.isnull().sum()
df.dtypes

df.columns
df.duplicated()
df[df.duplicated()] # there are no duplicates in data
df.shape
df.dtypes
df.info()
#boxplot
df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.5*IQR)
UW=Q3+(2.5*IQR)
df["age"]<LW
df[df["age"]<LW].shape
df["age"]>UW
df[df["age"]>UW].shape
df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))
#
df.boxplot("balance",vert=False)
Q1=np.percentile(df["balance"],25)
Q3=np.percentile(df["balance"],75)
IQR=Q3-Q1
LW=Q1-(2.5*IQR)
UW=Q3+(2.5*IQR)
df["balance"]<LW
df[df["balance"]<LW].shape
df["balance"]>UW
df[df["balance"]>UW].shape
df["balance"]=np.where(df["balance"]>UW,UW,np.where(df["balance"]<LW,LW,df["balance"]))
##
df.boxplot("day",vert=False)
##
###
df.dtypes
X=df[["age","balance","day","duration","campaign","pdays","previous"]]
X.dtypes
X.corr()
X.corr().to_csv("logicorr.csv")

#
X_new=df[["job","marital","education","default","housing","loan","contact","month","poutcome"]]

#data transform
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X=SS.fit_transform(X)
X=pd.DataFrame(X)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["job"]=LE.fit_transform(df["job"])
df["job"]=pd.DataFrame(df["job"])

df["marital"]=LE.fit_transform(df["marital"])
df["marital"]=pd.DataFrame(df["marital"])

df["education"]=LE.fit_transform(df["education"])
df["education"]=pd.DataFrame(df["education"])

df["default"]=LE.fit_transform(df["default"])
df["default"]=pd.DataFrame(df["default"])

df["housing"]=LE.fit_transform(df["housing"])
df["housing"]=pd.DataFrame(df["housing"])

df["loan"]=LE.fit_transform(df["loan"])
df["loan"]=pd.DataFrame(df["loan"])

df["contact"]=LE.fit_transform(df["contact"])
df["contact"]=pd.DataFrame(df["contact"])

df["month"]=LE.fit_transform(df["month"])
df["month"]=pd.DataFrame(df["month"])

df["poutcome"]=LE.fit_transform(df["poutcome"])
df["poutcome"]=pd.DataFrame(df["poutcome"])

X=pd.concat([X,df["job"],df["marital"],df["education"],df["default"],df["housing"],df["loan"],df["contact"],df["month"],df["poutcome"]],axis=1)
X
X.head()
Y=LE.fit_transform(df["y"])
Y=pd.DataFrame(Y)
#test and train
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,Y_train)
Y_predtrain=LR.predict(X_train)
Y_predtest=LR.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)
ac2=accuracy_score(Y_test,Y_predtest)
print(ac1)
print(ac2)

#Accuracy score is 89%....
from sklearn.metrics import log_loss
LL=log_loss(Y_train,Y_predtrain)

#log loss error is 3.....