# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:00:18 2022

@author: Dell
"""

import numpy as np
import pandas as pd

data=pd.read_csv("book.csv",encoding='latin-1')
data
data.columns
data=data.iloc[:,1:]
data.columns
data.shape
data.isnull().sum()
data.head()
data.duplicated
data=data.drop_duplicates(keep=False)  #duplicates are drop
data.shape

#sorting the user ID...
df=data.sort_values('User.ID')

df['User.ID'].unique()
len(df['User.ID'].unique())  #2182 user are there

df['Book.Title'].value_counts()
df["Book.Rating"].value_counts() #rating 8 is given 2283 times
df = data.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')

df=df.fillna(value=0,axis=0)
df.head()

#calculating cosine based similarties..
from sklearn.metrics import pairwise_distances
df=1-pairwise_distances(df.values,metric='cosine')
df
df=pd.DataFrame(df)
df

#changing the index name and columns name for better understanding the similarities btw the users....
df.index=data['User.ID'].unique()
df.columns=data['User.ID'].unique()
df.iloc[0:10,0:10]

#same user id has similarity as 1 so i have replace them with 0
np.fill_diagonal(df.values,0)
df

#checking the highest similarities btw the users
df.idxmax(axis=1)[0:20]

#checking which similarities the users have
data[(data['User.ID']==276737) | (data["User.ID"]==276726)]
