# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 23:11:25 2022

@author: mikeydrako123
"""

import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation

def separator(data, start ,strout):
    Y = data[strout]
    X = data.iloc[:,start:data.shape[1] - 1]
    return X, Y

data = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')

X, Y = separator(data, 0, 'Month')
x = X.to_numpy()
plt.scatter(x[:,0], x[:,1])
plt.show()

clustering = AffinityPropagation(random_state=5).fit(X)
labels = clustering.labels_
x = X.to_numpy()
plt.scatter(x[:,0], x[:,1], c=labels)
plt.show()  
