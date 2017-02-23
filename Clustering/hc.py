#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 07:16:36 2017

@author: imcbv
"""

# Hierarchical CLustering

# Reset variables
%reset -f

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
                
# Using dendogram to find optimal n of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('dendogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

# Fitting HC to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c="red",label="Careful")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c="blue",label="Standard")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c="green",label="Target")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c="cyan",label="Careless")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c="magenta",label="Sensible")
plt.title("Cluster of clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score") 
plt.legend()
plt.show()