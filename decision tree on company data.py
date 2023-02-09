# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:06:15 2022

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



#model fitting...

from sklearn.tree import DecisionTreeClassifier

#the model is overfitting so we are doing hyperparameter tuning

DT=DecisionTreeClassifier(max_depth=5,max_leaf_nodes=20)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)

Y_predtest=DT.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)

ac2=accuracy_score(Y_test,Y_predtest)


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

#the best values are max_depth is 5 and max_leaf_nodes is 20 then ac1=73% and ac2=55%


#if max_depth is 5 and max_leaf_nodes is 54 then ac1=77% and ac2=55%

#if max_depth is 5 and max_leaf_nodes is 15 then ac1=70% and ac2=55%

##if max_depth is 5 and max_leaf_nodes is 20 then ac1=73% and ac2=55%

#The model is over fittng.....


#bagging
from sklearn.ensemble import BaggingClassifier
DT=DecisionTreeClassifier(max_depth=5)
Bag=BaggingClassifier(base_estimator=DT,max_samples=0.7,n_estimators=100)
Bag.fit(X_train,Y_train)                     

Y_predtrain=Bag.predict(X_train)

Y_predtest=Bag.predict(X_test)

from sklearn.metrics import accuracy_score
ac1=accuracy_score(Y_train,Y_predtrain)

ac2=accuracy_score(Y_test,Y_predtest)


##if max_depth is 5 , max samples is 0.5 and n estimators is 100 then ac1=82% and ac2=60%

##if max_depth is 5 , max samples is 0.7 and n estimators is 100 then ac1=86% and ac2=57%


#===========================================================================
#model 2 using entropy

import pandas as pd 
import numpy as np

df=pd.read_csv("Company_Data.csv")
df.head()
df.dtypes
df.info()
df.duplicated()
df[df.duplicated()]

df.corr()

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

#model fitting...

from sklearn.tree import DecisionTreeRegressor

#the model is overfitting so we are doing hyperparameter tuning

DT=DecisionTreeRegressor(max_depth=5,max_leaf_nodes=20)
DT.fit(X_train,Y_train)
Y_predtrain=DT.predict(X_train)

Y_predtest=DT.predict(X_test)

from sklearn.metrics import r2_score
ac1=r2_score(Y_train,Y_predtrain)
#ac1=72%
ac2=r2_score(Y_test,Y_predtest)
#ac2=41%

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

#bagging

from sklearn.ensemble import BaggingRegressor
DT=DecisionTreeRegressor(max_depth=5)
Bag=BaggingRegressor(base_estimator=DT,max_samples=0.7,n_estimators=100)
Bag.fit(X_train,Y_train)                     

Y_predtrain=Bag.predict(X_train)

Y_predtest=Bag.predict(X_test)

from sklearn.metrics import r2_score
ac1=r2_score(Y_train,Y_predtrain)
#ac1=81%
ac2=r2_score(Y_test,Y_predtest)
#ac2=62%

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


#therefore after applying bagging ac1 is 81% and ac2 is 68% 