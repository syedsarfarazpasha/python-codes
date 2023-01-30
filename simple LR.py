# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:04:26 2022

@author: Dell
"""

import numpy as np 
import pandas as pd
df=pd.read_csv("delivery_time.csv")
df
df.head()
df.shape
df.isnull().sum() #there is no null values
df.describe()
df.boxplot("Delivery Time",vert=False)
df.boxplot("Sorting Time",vert=False) #there are no outliers


df.corr()     # both variables are strong positive correlated

df.plot.scatter(x="Delivery Time",y="Sorting Time") #if sorting time increses then delivery time also increses

df["Sorting Time"].hist()
df["Delivery Time"].hist()
#spliting
X=df[["Sorting Time"]]
Y=df["Delivery Time"]

#Model 1

#Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
y1=LR.predict(X)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,y1)
#mse=7.79

np.sqrt(mse).round(3)
#Rmse=2.79

from sklearn.metrics import r2_score
r2_score(Y,y1)
#rscore is 68%


#Model 2

#Model fitting with X**2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1=LR.predict(X**2)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,y1)

#mse=9.06

np.sqrt(mse).round(3)
#Rmse=3.011

from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 63%



#Model 3

#Model fitting with np.log(X)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1=LR.predict(np.log(X))

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,y1)

#mse=7.47

np.sqrt(mse).round(3)
#Rmse=2.73

from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 69%



#Model 4

#Model fitting with np.sqrt(X)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1=LR.predict(np.sqrt(X))

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,y1)

#mse=7.46

np.sqrt(mse).round(3)
#Rmse=2.73

from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 69%



