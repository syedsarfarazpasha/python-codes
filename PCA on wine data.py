# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:21:48 2022

@author: Dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("wine.csv")
df
df.shape
#droping the first column as per the question/case study....
df=df.drop("Type",axis=1) #type is a column where clustering is already done.
df
df.head()
df.shape
df.info()
df.duplicated()
df[df.duplicated()]



#Standardization.....
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
df=SS.fit_transform(df)
df=pd.DataFrame(df)


#PCA
from sklearn.decomposition import PCA
pca=PCA()
Y=pca.fit_transform(df)

percentage=pca.explained_variance_ratio_



final_data=pd.DataFrame(data=Y,columns=['P0C1', 'P0C2','P0C3','P0C4','P0C5','P0C6','P0C7','P0C8','P096','P0C10','P1C1', 'P1C2','P1C3'])
final_data
final_data.shape

#After PCA we are taking first three columns because the  more percentage of data is present in first three columns
X=final_data.iloc[:,0:3]
X
X.shape

#clustering....

########################   KMeans clustering ######################
#first checking that how many clusters are best based on inertia value diffrence..
from sklearn.cluster import KMeans
inertia=[]
for i in range(1,10):
    KM=KMeans(n_clusters=i,random_state=10)
    KM.fit(X)
    inertia.append(KM.inertia_)
print(inertia)


#Elbow method to see variance in inertia by clusters

plt.plot(range(1,10),inertia)
plt.title("Elbow Method")
plt.xlabel("No of clusters")
plt.ylabel("Inertia")
plt.show()

#Therefore the variance in inertia  b/w 3th and 4th cluster is less we can go with 3 clusters


#SCREE plot
f1={'clusters':range(1,10),'inertia values':inertia}
f2=pd.DataFrame(f1)
sns.barplot(x='clusters',y='inertia values',data=f2)

#therefore by scree plot we can tell that 3 clusters are best

KM=KMeans(n_clusters=3,n_init=30)
Y=KM.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()

new_data=pd.concat([X,Y],axis=1)
#apply any classifier techniques for making model


##########################   heirarchial clustering    ##############################
from sklearn.cluster import AgglomerativeClustering
AG=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
Y=AG.fit_predict(X)
Y=pd.DataFrame(Y)

Y.value_counts()

new_data=pd.concat([X,Y],axis=1)

#apply any classifier techniques for making model

#conclusion:The no of clusters i have obtained is same number of clusters with the original data  
#orinal data has 3 clusters
#i have also obtained  3 clusters... 
