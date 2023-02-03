# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:11:16 2022

@author: Dell
"""

import numpy as np
import pandas as pd
df=pd.read_csv("ToyotaCorolla.csv",encoding='latin-1')
df
df.head()
df.isnull().sum()
df.shape
df.duplicated()
df[df.duplicated()]
df=df.drop([113],axis=0) #duplicate is removed
df.shape
#splitting
Y=df["Price"]
X=df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
X.corr()
X.corr().to_csv("MLR.csv")
df=pd.concat([X,Y],axis=1)
df
df.corr()
df.corr().to_csv("MLR.csv ")

#X varibles are not dependent on each other

#splitting
Y=df["Price"]
X=df[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]


#boxplot
df.boxplot("Age_08_04",vert=False)
Q1=np.percentile(df["Age_08_04"],25)
Q3=np.percentile(df["Age_08_04"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR) #outliers are there so i have increase range 1.5 to 2.5
LW=Q1-(2.5*IQR)
df["Age_08_04"]<LW
df[df["Age_08_04"]<LW].shape

df.boxplot("KM",vert=False)
Q1=np.percentile(df["KM"],25)
Q3=np.percentile(df["KM"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df["KM"]>UW
df[df["KM"]>UW].shape
df["KM"]=np.where(df["KM"]>UW,UW,np.where(df["KM"]<LW,LW,df["KM"]))  #replacing the outliers with UW and LW values...
###
df.boxplot("HP",vert=False)
Q1=np.percentile(df["HP"],25)
Q3=np.percentile(df["HP"],75)
IQR=Q3-Q1
UW=Q3+(1.5*IQR)
LW=Q1-(1.5*IQR)
df["HP"]>UW
df[df["HP"]>UW].shape
df["HP"]=np.where(df["HP"]>UW,UW,np.where(df["HP"]<LW,LW,df["HP"]))
#
df.boxplot("cc",vert=False)
Q1=np.percentile(df["cc"],25)
Q3=np.percentile(df["cc"],75)
IQR=Q3-Q1
UW=Q3+(2.0*IQR)
LW=Q1-(2.0*IQR)
df["cc"]>UW
df[df["cc"]>UW].shape
df["cc"]=np.where(df["cc"]>UW,UW,np.where(df["cc"]<LW,LW,df["cc"]))

#outliers are 80.
df.boxplot("Doors",vert=False)
##
df.boxplot("Gears",vert=False)
Q1=np.percentile(df["Gears"],25)
Q3=np.percentile(df["Gears"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df["Gears"]>UW
df[df["Gears"]>UW].shape
df["Gears"]<LW
df[df["Gears"]<LW].shape
df["Gears"]=np.where(df["Gears"]>UW,UW,np.where(df["Gears"]<LW,LW,df["Gears"]))

##
df.boxplot("Quarterly_Tax",vert=False)
Q1=np.percentile(df["Quarterly_Tax"],25)
Q3=np.percentile(df["Quarterly_Tax"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df["Quarterly_Tax"]>UW
df[df["Quarterly_Tax"]>UW].shape
df["Quarterly_Tax"]<LW
df[df["Quarterly_Tax"]<LW].shape
df["Quarterly_Tax"]=np.where(df["Quarterly_Tax"]>UW,UW,np.where(df["Quarterly_Tax"]<LW,LW,df["Quarterly_Tax"]))
##
df.boxplot("Weight",vert=False)
Q1=np.percentile(df["Weight"],25)
Q3=np.percentile(df["Weight"],75)
IQR=Q3-Q1
UW=Q3+(2.5*IQR)
LW=Q1-(2.5*IQR)
df["Weight"]>UW
df[df["Weight"]>UW].shape
df["Weight"]<LW
df[df["Weight"]<LW].shape
df["Weight"]=np.where(df["Weight"]>UW,UW,np.where(df["Weight"]<LW,LW,df["Weight"]))
df.duplicated()
df[df.duplicated()]
#test and train
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)



#Model 1


#model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)
Y_predtrain=LR.predict(X_train)
Y_predtest=LR.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(Y_train,Y_predtrain)
score=r2_score(Y_test ,Y_predtest)

import statsmodels.api as sma
X_new = sma.add_constant(X)
lm2 = sma.OLS(Y,X).fit()
lm2.summary()

#p value of CC,Doors,Gears is >0.05 Therefore it has colinearityissues....



#Model 2

#In Model 2 we are removing Doors column....

#splitting
Y=df["Price"]
X1=df[["Age_08_04","KM","HP","cc","Gears","Quarterly_Tax","Weight"]]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.3)



#model fitting
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)
Y_predtrain=LR.predict(X_train)
Y_predtest=LR.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(Y_train,Y_predtrain)
score=r2_score(Y_test ,Y_predtest)

import statsmodels.api as sma
X_new = sma.add_constant(X1)
lm2 = sma.OLS(Y,X1).fit()
lm2.summary()

#Model 2 is best because R2 score is 85% and Pvalue of variables is <0.05


