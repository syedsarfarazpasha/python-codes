# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:08:08 2022

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#train data
df_train=pd.read_csv("SalaryData_Train.csv")
df_train
df_train.shape
df_train.info()

#test data

df_test=pd.read_csv("SalaryData_Test.csv")
df_test
df_test.shape
df_test.info()

#concating
df=pd.concat([df_train,df_test],axis=0)
df.shape
df.info()
df.corr()
df.corr().to_csv('naive.csv') #there is no relation btw independent variables
df.head()
#outliers dectection and treating outliers...
#
df.boxplot("age",vert=False)
Q1=np.percentile(df["age"],25)
Q3=np.percentile(df["age"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["age"]<LW
df[df["age"]<LW]
df[df["age"]<LW].shape

df["age"]>UW
df[df["age"]>UW]
df[df["age"]>UW].shape

df["age"]=np.where(df["age"]>UW,UW,np.where(df["age"]<LW,LW,df["age"]))

#spliting 
X=df.iloc[:,0:13]
X.columns
Y=df["Salary"]

#Standardization is not requird because naive bayes algorithm work on the principle of conditional probability....
#changing categorical into numerical

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


X["workclass"]=LE.fit_transform(X["workclass"])
X["education"]=LE.fit_transform(X["education"])
X["maritalstatus"]=LE.fit_transform(X["maritalstatus"])
X["occupation"]=LE.fit_transform(X["occupation"])
X["relationship"]=LE.fit_transform(X["relationship"])
X["race"]=LE.fit_transform(X["race"])
X["sex"]=LE.fit_transform(X["sex"])
X["native"]=LE.fit_transform(X["native"])
X

#test and train
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

#model fitting
from sklearn.naive_bayes import MultinomialNB
NB=MultinomialNB()
NB.fit(X_train,Y_train) 
Y_predtrain=NB.predict(X_train)
Y_predtest=NB.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)
AC2=accuracy_score(Y_test,Y_predtest)

print(AC1,AC2)

#Train Accuracy=77.28%
#Test Accuracy=77.52%

prediction=NB.predict(np.array([[36,0,2,13,0,0,1,1,1,0,0,40,16]]))


