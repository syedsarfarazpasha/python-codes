# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 11:23:09 2022

@author: Dell
"""

import numpy as np 
import pandas as pd
df=pd.read_csv("Salary_Data.csv")
df
df.head()
df.shape
df.isnull().sum() #there is no null values
df.describe()
df.boxplot("Salary",vert=False)
df.boxplot("YearsExperience",vert=False) #there are no outliers


df.corr()     # both variables are strong positive correlated

df.plot.scatter(x="Salary",y="YearsExperience") #if yearsExperience increses then salary also increses

df["YearsExperience"].hist()
df["Salary"].hist()
#spliting
X=df[["YearsExperience"]]
Y=df["Salary"]


#Model 1

#Model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
y1=LR.predict(X)



from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 95%


#Model 2

#Model fitting with X**2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X**2,Y)
y1=LR.predict(X**2)


from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 91%



#Model 3

#Model fitting with np.log(X)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.log(X),Y)
y1=LR.predict(np.log(X))


from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 85%



#Model 4

#Model fitting with np.sqrt(X)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(np.sqrt(X),Y)
y1=LR.predict(np.sqrt(X))


from sklearn.metrics import r2_score
r2_score(Y,y1)

#rscore is 93%

