# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:49:09 2020

@author: nitesh vamshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[: , [3 , 4]].values

# using the dendrogram to find optimal number of clusters

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrograms')
plt.xlabel('Customers')
plt.ylabel('Eculidean distance')
plt.show()

# Applying hierarchial clustering to the mall dataset

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc== 0,0],X[y_hc== 0,1],s=100,c='red',label='Cluster1')
plt.scatter(X[y_hc== 1,0],X[y_hc== 1,1],s=100,c='blue',label='Cluster2')              
plt.scatter(X[y_hc== 2,0],X[y_hc== 2,1],s=100,c='green',label='Cluster3')
plt.scatter(X[y_hc== 3,0],X[y_hc== 3,1],s=100,c='cyan',label='Cluster4')    
plt.scatter(X[y_hc== 4,0],X[y_hc== 4,1],s=100,c='magenta',label='Cluster5')
plt.title('Cluster of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.show() 