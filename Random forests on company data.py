# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:22:16 2022

@author: Dell
"""

import pandas as pd
import numpy as np
df=pd.read_csv("Company_Data.csv")
df.head()
df.dtypes
df.info()
df.duplicated()
df[df.duplicated()]

df.corr()
df.corr().to_csv("Dtree.csv")

#Boxplots

df.boxplot("CompPrice",vert=False)
Q1=np.percentile(df["CompPrice"],25)
Q3=np.percentile(df["CompPrice"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["CompPrice"]<LW].shape
df[df["CompPrice"]>UW].shape

df["CompPrice"]=np.where(df["CompPrice"]>UW,UW,np.where(df["CompPrice"]<LW,LW,df["CompPrice"]))
###
df.boxplot("Income",vert=False)
Q1=np.percentile(df["Income"],25)
Q3=np.percentile(df["Income"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["Income"]<LW].shape
df[df["Income"]>UW].shape

df["Income"]=np.where(df["Income"]>UW,UW,np.where(df["Income"]<LW,LW,df["Income"]))


######

df.boxplot("Advertising",vert=False)

####
df.boxplot("Population",vert=False)

#####
df.boxplot("Price",vert=False)
Q1=np.percentile(df["Price"],25)
Q3=np.percentile(df["Price"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df[df["Price"]<LW].shape
df[df["Price"]>UW].shape

df["Price"]=np.where(df["Price"]>UW,UW,np.where(df["Price"]<LW,LW,df["Price"]))

#######
df.boxplot("Age",vert=False)

#####
df.boxplot("Education",vert=False)

#coverting Sales into categorical...
df["Sales"] = pd.cut(df["Sales"], bins=[0,4.2,8.01,12.01,16.27],labels=["poor","good","very good","excellent"])

df

#spilting 
X=df.iloc[:,1:]
X.columns

X.dtypes

#Standidization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()


X["CompPrice"]=MM.fit_transform(X[["CompPrice"]])

X["Income"]=MM.fit_transform(X[["Income"]])

X["Advertising"]=MM.fit_transform(X[["Advertising"]])

X["Population"]=MM.fit_transform(X[["Population"]])

X["Price"]=MM.fit_transform(X[["Price"]])

X["Age"]=MM.fit_transform(X[["Age"]])

X["Education"]=MM.fit_transform(X[["Education"]])


X["ShelveLoc"]=LB.fit_transform(X["ShelveLoc"])
X["ShelveLoc"]=pd.DataFrame(X["ShelveLoc"])


X["Urban"]=LB.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["US"]=LB.fit_transform(X["US"])
X["US"]=pd.DataFrame(X["US"])

X
#Y variable..

df["Sales"]=LB.fit_transform(df["Sales"])
Y=pd.DataFrame(df["Sales"])

Y

#train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


#model fitting
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_depth=5,max_leaf_nodes=20)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)

AC2=accuracy_score(Y_test,Y_predtest)

#if max_depth is 10 and max_leaf_nodes is 20 then AC1=82% and ac2=61%

#if max_depth is 5 and max_leaf_nodes is 15 then AC1=74% and AC2=59%

##if max_depth is 5 and max_leaf_nodes is 20 then ac1=78% and ac2=57%

#=============================================================================
#model 2 using random forest regression 
df=pd.read_csv("Company_Data.csv")

Y=df["Sales"]

X=df.iloc[:,1:]
X.columns

X.dtypes

#Standidization

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

from sklearn.preprocessing import LabelEncoder
LB=LabelEncoder()


X["CompPrice"]=MM.fit_transform(X[["CompPrice"]])

X["Income"]=MM.fit_transform(X[["Income"]])

X["Advertising"]=MM.fit_transform(X[["Advertising"]])

X["Population"]=MM.fit_transform(X[["Population"]])

X["Price"]=MM.fit_transform(X[["Price"]])

X["Age"]=MM.fit_transform(X[["Age"]])

X["Education"]=MM.fit_transform(X[["Education"]])


X["ShelveLoc"]=LB.fit_transform(X["ShelveLoc"])
X["ShelveLoc"]=pd.DataFrame(X["ShelveLoc"])


X["Urban"]=LB.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

X["US"]=LB.fit_transform(X["US"])
X["US"]=pd.DataFrame(X["US"])

X

#train and test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


#model fitting

from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(max_depth=5,max_leaf_nodes=20)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import r2_score

r2score1=r2_score(Y_train,Y_predtrain)
#r2score1=80%

r2score2=r2_score(Y_test,Y_predtest)
#r2score2=68%

from sklearn.model_selection import GridSearchCV
import numpy as np

d1={'max_depth':np.arange(0,50,1),
     'max_leaf_nodes':np.arange(0,50,1)}

Gridgb=GridSearchCV(estimator=RandomForestRegressor(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_
