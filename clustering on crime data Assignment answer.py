# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:27:31 2022

@author: Dell
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("crime_data.csv")
df.head()
df.info()

X=df.iloc[:,1:]
X

df.boxplot("Murder",vert=False)
df.boxplot("Assault",vert=False)
df.boxplot("UrbanPop",vert=False)
#there is no outliers for above thre variables
df.boxplot("Rape",vert=False)
Q1=np.percentile(df["Rape"],25)
Q3=np.percentile(df["Rape"],75)
IQR=Q3-Q1
LW=Q1-(1.5*IQR)
UW=Q3+(1.5*IQR)
df["Rape"]>UW
df[df["Rape"]>UW]
df.drop([1,27],axis=0,inplace=True)
df.shape
X=df.iloc[:,1:]
X
#standardization of data
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X=SS.fit_transform(X)
X=pd.DataFrame(X)


##############################     Hierarchical clustering              ###########3#########
from sklearn.cluster import AgglomerativeClustering
AC=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='complete')
Y=AC.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()
## clustres are Out[127]: 
0    18
1    13
3    10
2     7
# In Hierarchical clustering we cannot decide hou many clusters are required but the best we are getting with 4 clusters...
new_data1=pd.concat([df,Y],axis=1)
#apply any classifier techniques for making model





#########################            K-Means clustering    ########################
#first we checking how many clusters are requried..
from sklearn.cluster import KMeans
inertia = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    inertia.append(km.inertia_)

print(inertia)

#Elbow method to see variance in inertia by clusters


plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia')
plt.show()

#Therefore the variance in inertia  b/w 4th and 5th cluster is less we can go with 4 clusters
#Scree plot to see variance in inertia by clusters
import seaborn as sns
d1 = {"kvalue": range(1, 11),'inertiavalues':inertia}
d2 = pd.DataFrame(d1)

import seaborn as sns
sns.barplot(x='kvalue',y="inertiavalues", data=d2)
#Therefore the variance in inertia  b/w 4th and 5th cluster is less we can go with 4 clusters

from sklearn.cluster import KMeans
KM=KMeans(n_clusters=4,n_init=10,max_iter=300)
Y=KM.fit_predict(X)
Y=pd.DataFrame(Y)
Y.value_counts()

# clustres are Out[122]: 
1    16
2    13
3    11
0     8
new_data2=pd.concat([df,Y],axis=1)
#apply any classifier techniques for making model



###########################                 DBSCAN                ###############################
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=1,min_samples=3)
dbscan.fit(X)
Y=dbscan.labels_
Y=pd.DataFrame(Y,columns=['cluster'])
df=pd.concat([df,Y],axis=1)
noisedata=df["cluster"]==-1
noisedata=df[df["cluster"]==-1]  #the output are outliers/noise data after removing this we can apply any other clustering techniques for better output


#therefore Kmeans is giving best results..

