from sklearn.model_selection import train_test_split #hold_out
from sklearn.model_selection import StratifiedKFold #stratified cross validation
from sklearn.model_selection import LeaveOneOut #LeaveOneOut
from sklearn.model_selection import KFold #Kfold
from sklearn.model_selection import ShuffleSplit

import numpy as np
import clasification as clf

def selected_datas(skms, X, Y, i):
    X = X.to_numpy()
    Y = Y.to_numpy()
    
    for train_index, test_index in skms.split(X, Y):
        X_train = np.stack(X[train_index])
        X_test = np.stack(X[test_index])    
        Y_train = np.stack(Y[train_index])
        Y_test = np.stack(Y[test_index])   

    return X_train, X_test, Y_train, Y_test
        
def hold_out(X, Y, train_size, test_size):
    #X_train, X_test, Y_train, Y_test
    if((train_size == 1.0) or (test_size == 1.0)):
        return X, X, Y, Y
    return train_test_split(X, Y, train_size= train_size, test_size=test_size) 

def random_subsampling(X, Y, n_splits, train, test, state):
    rs = ShuffleSplit(n_splits=n_splits, train_size=train , test_size=test, random_state=state)
    rs.get_n_splits(X, Y)
    return selected_datas(rs, X, Y, 0)

def K_fold(X, Y, n_splits, state, shuffle):
    kf = KFold(n_splits=n_splits, random_state = state, shuffle=shuffle)
    kf.get_n_splits(X, Y)
    return selected_datas(kf, X, Y, 1)
        
def stratified_cross_validation(X, Y, n_splits):
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, Y)
    return selected_datas(skf, X, Y, 3)

def prossesing(X, Y, cv, classifier):
    Y_preds = np.empty((Y.shape[0]), dtype = (str))

    for train_index, test_index in cv.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train = Y[train_index]
        if (classifier == 0):
            Y_pred = clf.sgd(X_train, Y_train, X_test)
        if (classifier == 1):
            Y_pred = clf.decision_Tree(X_train, Y_train, X_test)
        if (classifier == 2):
            Y_pred = clf.vector_machine(X_train, Y_train, X_test)
        if (classifier == 3):
            Y_pred = clf.neighbors(X_train, Y_train, X_test) 
        if (classifier == 4):
            Y_pred = clf.NNmodels(X_train, Y_train, X_test)
            
        Y_pred = Y_pred.ravel()
        Y_preds[train_index] = Y_pred[0]
    
    return Y_preds

def leave_one_out(X, Y):
    cv = LeaveOneOut()
    cv.get_n_splits(X, Y)
    X = X.to_numpy()
    Y = Y.to_numpy()
    Y_preds0 = prossesing(X, Y, cv, 0)
    Y_preds1 = prossesing(X, Y, cv, 1)
    Y_preds2 = prossesing(X, Y, cv, 2)
    Y_preds3 = prossesing(X, Y, cv, 3)
    Y_preds4 = prossesing(X, Y, cv, 4)

    return Y_preds0, Y_preds1, Y_preds2, Y_preds3, Y_preds4