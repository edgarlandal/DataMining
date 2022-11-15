# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 00:18:15 2022

@author: mikeydrako123
"""

import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering

def separator(data, start ,strout):
    Y = data[strout]
    X = data.iloc[:,start:data.shape[1] - 1]
    return X, Y

data = pd.read_csv('iris.csv')
X, Y = separator(data, 0, 'species')
x = X.to_numpy()
plt.scatter(x[:,0], x[:,1])
plt.show()

sc = SpectralClustering(n_clusters=4).fit(x)
print(sc)
labels = sc.labels_
x = X.to_numpy()
plt.scatter(x[:,0], x[:,1], c=labels)
plt.show()  
