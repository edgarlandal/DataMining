from sklearn.model_selection import train_test_split #hold_out
from sklearn.model_selection import KFold #Kfold
from sklearn.model_selection import ShuffleSplit

import numpy as np

def selected_datas(skms, X, Y):
    for train_index, test_index in skms.split(X):
        X_train = np.stack(X[train_index])
        X_test = np.stack(X[test_index])    
        Y_train = np.stack(Y[train_index])
        Y_test = np.stack(Y[test_index])   

    return X_train, X_test, Y_train, Y_test
        
def hold_out(X, Y, train_size, test_size):
    return train_test_split(X, Y, train_size= train_size, test_size=test_size) 

def random_subsampling(X, Y, n_splits, train, test, state):
    rs = ShuffleSplit(n_splits=n_splits, train_size=train , test_size=test, random_state=state)
    rs.get_n_splits(X, Y)
    return selected_datas(rs, X, Y)

def K_fold(X, Y, n_splits, state, shuffle):
    kf = KFold(n_splits=n_splits, random_state = state, shuffle=shuffle)
    kf.get_n_splits(X, Y)
    return selected_datas(kf, X, Y)