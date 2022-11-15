import pandas as pd
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation 

def separator(data ,strout):
    Y = data[strout]
    X = data.drop(strout, axis = 1)
    return X, Y

def separator_for_index(data):
    Y = data.iloc[:,-1]
    X = data.iloc[:,0:data.shape[1] - 1]
    return X, Y

def evalution(y, y_train):
    score = metrics.homogeneity_score(y, y_train)
    print("Homogeneity Score = {}".format(score))
    score = metrics.completeness_score(y, y_train)
    print("Completeness Score = {}".format(score))
    score = metrics.rand_score(y, y_train)
    print("Rand Score = {}".format(score))
    
def get_data_clustering_agglomerative(data):    
    x = data.to_numpy()
    aggloclust=AgglomerativeClustering().fit_predict(x)

    return x, aggloclust

def get_data_clustering_Affinity(data):    
    x = data.to_numpy()
    sc = AffinityPropagation().fit_predict(x)
    return x, sc

data = pd.read_csv('zoo.csv')

print(data.head())
print(data.describe())
print(data.info())

data = data.drop('aardvark', axis = 1)

x, y = separator_for_index(data);
_, y_leb = get_data_clustering_agglomerative(data)
evalution(y, y_leb)
_, y_leb = get_data_clustering_Affinity(data)
evalution(y, y_leb)

def separator(data, n_inputs, n_targets):
    inputs = data.iloc[:,0:n_inputs]
    targets = data.iloc[:,n_inputs:(n_targets + n_inputs)]
    return inputs, targets