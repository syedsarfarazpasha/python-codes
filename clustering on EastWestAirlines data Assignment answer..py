# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 18:40:49 2022

@author: Dell
"""

import pandas as pd 
import numpy as np
df=pd.read_excel("EastWestAirlines.xlsx",sheet_name='data')
df.head()
df.info()
#we don't need ID column so we can remove
df.drop(columns=["ID#"],axis=1,inplace=True)
df.info()
df.shape
df.isnull().sum()
#Standardization the data
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
Y=SS.fit_transform(df)
X_new=pd.DataFrame(Y) 




####################################          KMeans clustering            ######################
#first we deciding how many cluster are required...
from sklearn.cluster import KMeans
KM=KMeans()
Inertia=[]
for i in range(1,10):
    KM=KMeans(n_clusters=i,random_state=0)
    KM.fit(X_new)
    inrt=KM.inertia_
    Inertia.append(inrt)
    
print(Inertia)    

#scree plot
import seaborn as sns
d1 = {"kvalue": range(1, 10),'inertiavalues':Inertia}
d2 = pd.DataFrame(d1)

sns.barplot(x='kvalue',y="inertiavalues", data=d2)
#Therefore the variance in inertia  b/w 3th and 4th cluster is less we can go with 3 clusters
# so by plot we can use 3 or 4 cluster but  i am using no of cluster is equal to 3
KM=KMeans(n_clusters=3,n_init=30)
X=KM.fit_predict(X_new)
Y=pd.DataFrame(X,columns=['cluster'])
Y.value_counts()
new_data=pd.concat([df,Y],axis=1)

#clusters are Out[154]: 
cluster
0          2574
1          1261
2           164 
#apply any classifier techniques to new_data for making model



#######################        Hierarchical clustering        #####################################
from sklearn.cluster import AgglomerativeClustering
AG=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
Y=AG.fit_predict(X_new)
Y=pd.DataFrame(Y)
Y.value_counts()
new_data2=pd.concat([df,Y],axis=1)
#clusters are Out[160]: 
0    3980
2      15
1       4
#apply any classifier techniques to new_data for making model



######################## DBSCAN ###########################
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=1,min_samples=3)
dbscan.fit_predict(X_new)
Y=dbscan.labels_
Y=pd.DataFrame(Y,columns=["cluster"])
Y.value_counts()
final_data=pd.concat([df,Y],axis=1) 
noisedata=final_data[final_data["cluster"]==-1] #the output are outliers/noise data after removing this we can apply any other clustering techniques for better output


#therefore Kmeans is giving best results..
 

