# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:13:06 2020

@author: nitesh vamshi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[: , [3 , 4]].values

# using the elbow method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Wcss')
plt.show()

# Applying Kmeans to the mall dataset
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_means=kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_means== 0,0],X[y_means== 0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_means== 1,0],X[y_means== 1,1],s=100,c='blue',label='Cluster2')              
plt.scatter(X[y_means== 2,0],X[y_means== 2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_means== 3,0],X[y_means== 3,1],s=100,c='cyan',label='Cluster4')    
plt.scatter(X[y_means== 4,0],X[y_means== 4,1],s=100,c='magenta',label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()    

