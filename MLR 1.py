# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:24:06 2022

@author: Dell
"""

import numpy as np 
import pandas as pd
df=pd.read_csv("50_Startups.csv")
df
df.shape
df.isnull().sum()
df.dtypes

#boxplot
df.boxplot("R&D Spend",vert=False)
df.boxplot("Administration",vert=False)
df.boxplot("Marketing Spend",vert=False)

#there are no outliers...
X=df.iloc[:,0:4]  
X.corr()
#R&D spend and Marketing have some relation...


#changing categorical variable into int

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["State"]=LE.fit_transform(df["State"])
df["State"]=pd.DataFrame(df["State"])


#splitting


X=df.iloc[:,0:4]  
Y=df["Profit"]
X.corr()

#Model 1


#model fitting i.e model 1
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
Y_pred=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)

from sklearn.metrics import r2_score
score=r2_score(Y,Y_pred)

import statsmodels.api as sma
X_new = sma.add_constant(X)
lm2 = sma.OLS(Y,X).fit()
lm2.summary()

#################################

#Model 2

X1=df[["R&D Spend","Administration","State"]]

#model fitting i,e model 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X1,Y)
Y_pred=LR.predict(X1)


from sklearn.metrics import r2_score
score=r2_score(Y,Y_pred)

import statsmodels.api as sma
X_new = sma.add_constant(X1)
lm2 = sma.OLS(Y,X1).fit()
lm2.summary()


#Model 1 R2score is 95% 

#Model 2  R2score is  94%


# I preferred model 1 because R2score is good and p value is <0.05


predictions=LR.predict(np.array([[165349,136897,471784,2]]))
