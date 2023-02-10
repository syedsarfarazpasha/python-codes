# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 21:19:19 2022

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


#model fitting....
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier(max_depth=5)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)

Y_predtest=DT.predict(X_test)

from sklearn.metrics import accuracy_score

AC1=accuracy_score(Y_train,Y_predtrain)

AC2=accuracy_score(Y_test,Y_predtest)


#if max_depth is 5 then AC1=81% and AC2=77%


#To know which is the best max depth value and max leaf node value we are doing gridesearchcv.... 

from sklearn.model_selection import GridSearchCV
import numpy as np

d1={'max_depth':np.arange(0,100,1),
     'max_leaf_nodes':np.arange(0,100,1)}

Gridgb=GridSearchCV(estimator=DecisionTreeClassifier(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_

#if max_depth is 3 and max_leaf_nodes is 5 then AC1=78% and AC2=82%


#========================================================================
#model 2 using entropy

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
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)


#model fitting....

from sklearn.tree import DecisionTreeRegressor
DT=DecisionTreeRegressor(max_depth=5)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)

Y_predtest=DT.predict(X_test)

from sklearn.metrics import r2_score

AC1=r2_score(Y_train,Y_predtrain)
AC1
AC2=r2_score(Y_test,Y_predtest)
AC2

from sklearn.model_selection import GridSearchCV
import numpy as np

d1={'max_depth':np.arange(0,100,1),
     'max_leaf_nodes':np.arange(0,100,1)}

Gridgb=GridSearchCV(estimator=DecisionTreeRegressor(),
                    param_grid=d1,
                    scoring=None)
Gridgb.fit(X_train,Y_train)
Gridgb.best_score_
Gridgb.best_params_

########################################################

Training_accuracy = []
Test_accuracy = []

for i in range(1,12):
    regressor = DecisionTreeRegressor(max_depth=i,criterion="squared_error") 
    regressor.fit(X_train,Y_train)
    Y_pred_train = regressor.predict(X_train)
    Y_pred_test = regressor.predict(X_test)
    Training_accuracy.append(r2_score(Y_train,Y_pred_train))
    Test_accuracy.append(r2_score(Y_test,Y_pred_test))
    
    
pd.DataFrame(Training_accuracy)
pd.DataFrame(Test_accuracy)

pd.concat([pd.DataFrame(range(1,12)) ,pd.DataFrame(Training_accuracy),pd.DataFrame(Test_accuracy)],axis=1)    

#best Training_accuracy = 73%

#best Test_accuracy= 73%
