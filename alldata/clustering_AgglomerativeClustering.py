import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

def evalution(x, labels):
    score = metrics.silhouette_score(x, labels)
    print("Silhouette Score = {}".format(score))
    score = metrics.davies_bouldin_score(x, labels)
    print("Davies Bouldin Score = {}".format(score))
    score = metrics.calinski_harabasz_score(x, labels)
    print("Calinski Harabasz Score = {}".format(score))

def plot_data_ungrouped(data):    
    x = data.to_numpy()
    plt.scatter(x[:,0], x[:,2])
    plt.show()

def plot_data_clustering(data):    
    x = data.to_numpy()
    aggloclust=AgglomerativeClustering().fit_predict(x)
    plt.scatter(x[:,0], x[:,1], c=aggloclust, cmap = 'rainbow')
    plt.show()
    return x, aggloclust

data = pd.read_csv('Pokemon.csv')

print(data.head())
print(data.describe())
print(data.info())

data['type1'] = data['type1'].factorize()[0]
data['type2'] = data['type2'].factorize()[0]

data = data.drop('name', axis = 1)

plot_data_ungrouped(data)
x, labels = plot_data_clustering(data)
evalution(x, labels)
