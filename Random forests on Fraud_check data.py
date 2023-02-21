# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:11:17 2022

@author: Dell
"""

import pandas as pd
import numpy as np

df=pd.read_csv("Fraud_check.csv")
df
df.isnull().sum()
df.info()

#boxplots
df.boxplot("City.Population",vert=False)

df.boxplot("Work.Experience",vert=False)
#There is no outliers

df["Taxable.Income"]=pd.cut(df["Taxable.Income"], bins=[0,30000,99620],labels=["Risky","Good"])

df

#spilting
Y=df["Taxable.Income"]

X1=df.iloc[:,:2]

X2=df.iloc[:,3:]

X=pd.concat([X1,X2],axis=1)
X.dtypes
#Standidization
from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["Undergrad"]=LE.fit_transform(X["Undergrad"])
X["Undergrad"]=pd.DataFrame(X["Undergrad"])


X["Marital.Status"]=LE.fit_transform(X["Marital.Status"])
X["Marital.Status"]=pd.DataFrame(X["Marital.Status"])


X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])

Y=LE.fit_transform(df["Taxable.Income"])
Y=pd.DataFrame(Y)


X["City.Population"]=MM.fit_transform(X[["City.Population"]])

X["Work.Experience"]=MM.fit_transform(X[["Work.Experience"]])

X

#train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#model fitting...
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(max_depth=6,max_leaf_nodes=15)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import accuracy_score
AC1=accuracy_score(Y_train,Y_predtrain)

AC2=accuracy_score(Y_test,Y_predtest)

#if max_depth is 10 and max_leaf_nodes is 15 then AC1=75% and AC2=55%

#if max_depth is 6 and max_leaf_nodes is 15 then AC1=74% and AC2=54%

#=========================================================================

# model 2 using random forest regressor

import pandas as pd
import numpy as np

df=pd.read_csv("Fraud_check.csv")
df
df.isnull().sum()
df.info()

Y=df["Taxable.Income"]

X1=df.iloc[:,:2]

X2=df.iloc[:,3:]

X=pd.concat([X1,X2],axis=1)
X.dtypes

#Standidization

from sklearn.preprocessing import MinMaxScaler
MM=MinMaxScaler()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

X["Undergrad"]=LE.fit_transform(X["Undergrad"])
X["Undergrad"]=pd.DataFrame(X["Undergrad"])


X["Marital.Status"]=LE.fit_transform(X["Marital.Status"])
X["Marital.Status"]=pd.DataFrame(X["Marital.Status"])


X["Urban"]=LE.fit_transform(X["Urban"])
X["Urban"]=pd.DataFrame(X["Urban"])


X["City.Population"]=MM.fit_transform(X[["City.Population"]])

X["Work.Experience"]=MM.fit_transform(X[["Work.Experience"]])

X

#train and test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#model fitting...

from sklearn.ensemble import RandomForestRegressor
RF=RandomForestRegressor(max_depth=5,max_leaf_nodes=20)
RF.fit(X_train,Y_train)
Y_predtrain=RF.predict(X_train)
Y_predtest=RF.predict(X_test)

from sklearn.metrics import r2_score
r2score1=r2_score(Y_train,Y_predtrain)
r2score1

r2score2=r2_score(Y_test,Y_predtest)
r2score2

from sklearn.model_selection import GridSearchCV
import numpy as np

d1={'max_depth':np.arange(0,100,1),
     'max_leaf_nodes':np.arange(0,100,1)}

Gridgb=GridSearchCV(estimator=RandomForestRegressor(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_



