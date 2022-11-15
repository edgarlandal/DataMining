from ReliefF import ReliefF
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

def separator(data, start ,strout):
    Y = data[strout]
    X = data.iloc[:,start:data.shape[1] - 1]
    return X, Y

def clasificator(Y):
    return Y.factorize()
            
def relief(data,X, Y, no_of_features):
    fs = ReliefF(n_neighbors=20, n_features_to_keep=no_of_features)
    X_train = fs.fit_transform(X, Y)
    print("(No. of tuples, No. of Columns before ReliefF) : "+str(data.shape)+
          "\n(No. of tuples, No. of Columns after ReliefF) : "+str(X_train.shape))    
    return X_train

def get_colums_selected(d, X):
    n = []
    for i in range(X.shape[1]):
        for j in range(d.shape[1]):
            if(np.array_equal(d.iloc[:,j].to_numpy(), X[:,i])):
                n.append(d.columns[j])
                break
    print(n)        
data = pd.read_csv('iris.csv')
X, Y  = separator(data,0 ,'species')
a = np.zeros(Y.shape[0])
a = clasificator(Y)
X_train1 = relief(data,X.to_numpy(), a[0], 3)
get_colums_selected(data, X_train1)

data2 = pd.read_csv('2015.csv')
X, Y  = separator(data2, 3,'Region')
a = clasificator(Y)
X_train2 = relief(data2,X.to_numpy(),a[0],4)
get_colums_selected(data2, X_train2)
