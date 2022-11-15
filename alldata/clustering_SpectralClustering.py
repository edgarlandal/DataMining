import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation 
from sklearn import metrics

def evaluetion(x, labels):
    score = metrics.silhouette_score(x, labels)
    print("Silhouette Score = {}".format(score))
    score = metrics.davies_bouldin_score(x, labels)
    print("Davies Bouldin Score = {}".format(score))
    score = metrics.calinski_harabasz_score(x, labels)
    print("Calinski Harabasz Score = {}".format(score))

def plot_data_ungrouped(data):    
    x = data.to_numpy()
    plt.scatter(x[:,0], x[:,1])
    plt.show()

def plot_data_clustering(data):    
    x = data.to_numpy()
    sc = AffinityPropagation().fit_predict(x)
    plt.scatter(x[:,0], x[:,1], c=sc)
    plt.show()
    return x, sc

data = pd.read_csv('iris.csv')

print(data.describe())
print(data.info())

plot_data_ungrouped(data)
x , labels = plot_data_clustering(data)
evaluetion(x, labels)