# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 10:59:29 2022

@author: Dell
"""

import numpy as np
import pandas as pd

df=pd.read_csv("forestfires.csv")
df.head()

df.isnull().sum()

df.duplicated()
df[df.duplicated()]
df.drop_duplicates(inplace=True)

df.columns
df.dtypes


#boxplot
df.boxplot("FFMC",vert=False)
Q1=np.percentile(df["FFMC"],25)
Q3=np.percentile(df["FFMC"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["FFMC"]<LW
df[df["FFMC"]<LW]
df[df["FFMC"]<LW].shape

df["FFMC"]>UW
df[df["FFMC"]>UW]
df[df["FFMC"]>UW].shape

df["FFMC"]=np.where(df["FFMC"]>UW,UW,np.where(df["FFMC"]<LW,LW,df["FFMC"]))


#
df.boxplot("DMC",vert=False)
Q1=np.percentile(df["DMC"],25)
Q3=np.percentile(df["DMC"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["DMC"]<LW
df[df["DMC"]<LW]
df[df["DMC"]<LW].shape

df["DMC"]>UW
df[df["DMC"]>UW]
df[df["DMC"]>UW].shape

df["DMC"]=np.where(df["DMC"]>UW,UW,np.where(df["DMC"]<LW,LW,df["DMC"]))

#
df.boxplot("DC",vert=False)
Q1=np.percentile(df["DC"],25)
Q3=np.percentile(df["DC"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["DC"]<LW
df[df["DC"]<LW]
df[df["DC"]<LW].shape

df["DC"]>UW
df[df["DC"]>UW]
df[df["DC"]>UW].shape

df["DC"]=np.where(df["DC"]>UW,UW,np.where(df["DC"]<LW,LW,df["DC"]))

#
df.boxplot("ISI",vert=False)
Q1=np.percentile(df["ISI"],25)
Q3=np.percentile(df["ISI"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["ISI"]<LW
df[df["ISI"]<LW]
df[df["ISI"]<LW].shape
df["ISI"]>UW
df[df["ISI"]>UW]
df[df["ISI"]>UW].shape
df["ISI"]=np.where(df["ISI"]>UW,UW,np.where(df["ISI"]<LW,LW,df["ISI"]))
#
df.boxplot("temp",vert=False)
Q1=np.percentile(df["temp"],25)
Q3=np.percentile(df["temp"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["temp"]<LW
df[df["temp"]<LW]
df[df["temp"]<LW].shape
df["temp"]>UW
df[df["temp"]>UW]
df[df["temp"]>UW].shape
df["temp"]=np.where(df["temp"]>UW,UW,np.where(df["temp"]<LW,LW,df["temp"]))

#
df.boxplot("RH",vert=False)
df.boxplot("RH",vert=False)
Q1=np.percentile(df["RH"],25)
Q3=np.percentile(df["RH"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["RH"]<LW
df[df["RH"]<LW]
df[df["RH"]<LW].shape
df["RH"]>UW
df[df["RH"]>UW]
df[df["RH"]>UW].shape
df["RH"]=np.where(df["RH"]>UW,UW,np.where(df["RH"]<LW,LW,df["RH"]))
#
df.boxplot("wind",vert=False)
Q1=np.percentile(df["wind"],25)
Q3=np.percentile(df["wind"],75)
IQR=Q3-Q1
LW=Q1-(2.0*IQR)
UW=Q3+(2.0*IQR)
df["wind"]<LW
df[df["wind"]<LW]
df[df["wind"]<LW].shape
df["wind"]>UW
df[df["wind"]>UW]
df[df["wind"]>UW].shape
df["wind"]=np.where(df["wind"]>UW,UW,np.where(df["wind"]<LW,LW,df["wind"]))

##
df.boxplot("wind",vert=False)

A=df.iloc[:,2:30]
A.columns
A.corr()
A.corr().to_csv("svm.csv") #there is no relation btw independent variables



#Spliting
X=df.iloc[:,:30]
X.columns
Y=df["size_category"]
X.dtypes
#Standardization 
#changing categorical into numerical

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["FFMC"]=MM.fit_transform(X[["FFMC"]])
X["DMC"]=MM.fit_transform(X[["DMC"]])
X["DC"]=MM.fit_transform(X[["DC"]])
X["ISI"]=MM.fit_transform(X[["ISI"]])
X["temp"]=MM.fit_transform(X[["temp"]])
X["RH"]=MM.fit_transform(X[["RH"]])
X["wind"]=MM.fit_transform(X[["wind"]])
X["rain"]=MM.fit_transform(X[["rain"]])
X["area"]=MM.fit_transform(X[["area"]])


X["month"]=LE.fit_transform(X["month"])
X["month"]=pd.DataFrame(X["month"])

X["day"]=LE.fit_transform(X["day"])
X["day"]=pd.DataFrame(X["day"])

df["size_category"]=LE.fit_transform(df["size_category"])
Y=pd.DataFrame(df["size_category"])
Y

X

#train and test... 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)



#model fitting
from sklearn.svm import SVC
# clf = SVC()
clf = SVC(kernel='linear',C=5.0)
#clf = SVC(kernel='poly',degree=3)
#clf = SVC(kernel='rbf', gamma=2)
clf.fit(X_train,Y_train)
Y_pred_train = clf.predict(X_train)
Y_pred_test = clf.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_pred_train)

AC2=accuracy_score(Y_test,Y_pred_test)


#for clf = SVC(kernel='linear',C=5.0)

#AC1=78%

#AC2=72%




#for clf = SVC(kernel='linear',C=2.0)

#AC1=76%

#AC2=70%


#clf = SVC(kernel='rbf', gamma=2)

#AC1=81%

#AC2=69%

