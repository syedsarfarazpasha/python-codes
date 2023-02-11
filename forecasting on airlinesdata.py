# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:23:18 2022

@author: Dell
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel("Airlines+Data.xlsx")
df
df.shape

df["year"]=df.Month.dt.strftime("%Y")
df["month"]=df.Month.dt.strftime("%b")
df
#getting dummies
df1=pd.get_dummies(df["month"])

#creating t variable ie no of months
lst=list(range(1,97))
df2=pd.DataFrame(lst,columns=["t"])

#square of t
lst=list(range(1,97))
df3=pd.DataFrame(lst,columns=["t_square"])

df4=df3**2.


X1=df["Passengers"]
df["log_Passengers"]=np.log(X1)
df


#concating

df=pd.concat([df,df1,df2,df4],axis=1)
df
df.columns

#plots

df["Passengers"].plot()

df.boxplot(vert=False)

df.hist()

#spliting
Train=df.head(88)
Test=df.tail(8)

#Models

#Linear Model

import statsmodels.formula.api as smf
linear_model=smf.ols('Passengers~year',data=Train).fit()
pred_linear=pd.Series(linear_model.predict(pd.DataFrame(Test['year'])))
rmse_linear=np.sqrt(np.mean(np.array(Test['Passengers'])-np.array(pred_linear))**2)
rmse_linear

#rmse=45.75

#Exponential 

import statsmodels.formula.api as smf
EXP=smf.ols('log_Passengers~year',data=Train).fit()
pred_EXP=pd.Series(EXP.predict(Test['year']))
rmse_EXP=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_EXP)))**2))
rmse_EXP

#rmse=66.63

#Quadratic 

Quadratic = smf.ols("Passengers~t+t_square",data=Train).fit()
pred_quad=pd.Series(Quadratic.predict(Test[["t","t_square"]]))
rmse_quad=np.sqrt(np.mean((np.array(Test["Passengers"])-np.array(pred_quad))**2))
rmse_quad
#rmse=58.30

#Additive seasonality

add_sea=smf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
pred_add_sea=pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmse_add_sea=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea)**2)))
rmse_add_sea

# Additive seasonality Quadratic
add_sea_Quadratic=smf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
pred_add_sea_Quadratic=pd.Series(add_sea_Quadratic.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]))
rmse_add_sea_Quadratic=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_Quadratic)**2)))
rmse_add_sea_Quadratic

#multiplicative seasonability
Mul_sea=smf.ols("log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
pred_Mul_sea=pd.Series(Mul_sea.predict(Test))
rmse_Mul_sea=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mul_sea)))**2))
rmse_Mul_sea

#rmse=148.45577274025464

#multiplicative addictive seasonability
Mul_add_sea=smf.ols("log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
pred_Mul_add_sea=pd.Series(Mul_add_sea.predict(Test))
rmse_Mul_add_sea=np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mul_sea)))**2))
rmse_Mul_add_sea
#rmse=148


#comparing the results

data ={"MODEL":pd.Series(["rmse_linear","rmse_EXP","rmse_quad","rmse_add_sea","rmse_add_sea_Quadratic","rmse_Mul_sea","rmse_Mul_add_sea"])}
data1 ={"MODEL":pd.Series([rmse_linear,rmse_EXP,rmse_quad,rmse_add_sea,rmse_add_sea_Quadratic,rmse_Mul_sea,rmse_Mul_add_sea])}

a=pd.DataFrame(data)
b=pd.DataFrame(data1)
new=pd.concat([a,b],axis=1)
new


#Best rmse value we are getting from Quadratic model i,e rmse_quad : 58.30476227446929



model_full =  smf.ols("Passengers~t+t_square",data=df).fit()

df.dtypes
df.drop(df.columns[[0,2,1,4]],axis=1,inplace=True)
df.columns

pred_new =pd.Series(model_full .predict(df))
df["new_Passengers"]=pd.Series(pred_new)
df
df.columns
df["new_Passengers"]
