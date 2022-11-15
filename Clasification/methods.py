import pull_apart as pa
import clasification as clf
import metrics as m

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay

def get_medition(cm):    
    return cm[0][0] , cm[0][1], cm[1][0], cm[1][1], cm[0][0] + cm[0][1], cm[1][0] + cm[1][1]

def method_metrics(Y_true, Y_pred, tipo):
    cm = m.matriz_confusion(Y_true, Y_pred)
    TP, FN, FP, TN, P, N = get_medition(cm)
    print("| "+ tipo + " | {:.8f} | {:.8f} | {:.8f} | {:.8f} | {:.8f}  |".format(m.accuracy(TP, TN, P, N), 
                                                                                 m.error_rate(FP, FN, P, N), m.presicion(TP, FP), 
                                                                                 m.recall(TP, FN), m.sensitivity(TN, P)))
    print("|-----------|------------|------------|------------|------------|-------------|")
    return cm

def plot_matriz_confusion(cm0 ,cm1, cm2, cm3, cm4):
    f, axes = plt.subplots(1, 5, figsize=(20, 5), sharey='row')
    ConfusionMatrixDisplay(cm0).plot(ax=axes[0], xticks_rotation=45)
    ConfusionMatrixDisplay(cm1).plot(ax=axes[1], xticks_rotation=45)
    ConfusionMatrixDisplay(cm2).plot(ax=axes[2], xticks_rotation=45)
    ConfusionMatrixDisplay(cm3).plot(ax=axes[3], xticks_rotation=45)
    ConfusionMatrixDisplay(cm4).plot(ax=axes[4], xticks_rotation=45)
    
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    plt.show()
    
def print_header():
    print("|-----------|------------|------------|------------|------------|-------------|")
    print("| Tipo      | Accuracy   | Error Rate | Presicion  | Recall     | Sensitivity |")
    print("|-----------|------------|------------|------------|------------|-------------|")
    
def method_classification(X_train, X_test, Y_train, Y_test):
    if (type(Y_train) != np.ndarray):
        Y_train = Y_train.to_numpy().ravel()
    else:
        Y_train = Y_train.ravel()
        
    if (type(Y_test) != np.ndarray):
        Y_test = Y_test.to_numpy().ravel()
    else:
        Y_test = Y_test.ravel()
            
    if (type(X_train) != np.ndarray):
        X_train = X_train.to_numpy()
        
    if (type(X_test) != np.ndarray):
        X_test = X_test.to_numpy()
            
    print_header()
       
    print(Y_train)

    Y_pred0 = clf.sgd(X_train, Y_train, X_test)
    Y_pred0 = Y_pred0.ravel()
    cm0 = method_metrics(Y_test, Y_pred0, "SGD      ")
    
    Y_pred = clf.decision_Tree(X_train, Y_train, X_test)
    Y_pred = Y_pred.ravel()
    cm1 = method_metrics(Y_test, Y_pred, "Trees    ")
    
    Y_pred = clf.vector_machine(X_train, Y_train, X_test)
    Y_pred = Y_pred.ravel()
    cm2 = method_metrics(Y_test, Y_pred, "Vector M ")
    
    Y_pred = clf.neighbors(X_train, Y_train, X_test) 
    Y_pred = Y_pred.ravel()
    cm3 = method_metrics(Y_test, Y_pred, "Neighbors")  
    
    Y_pred = clf.NNmodels(X_train, Y_train, X_test)
    Y_pred = Y_pred.ravel()
    cm4 = method_metrics(Y_test, Y_pred, "NNModels ")
    
    plot_matriz_confusion(cm0, cm1, cm2, cm3, cm4)
    

def method_hold_out(X, Y, ts, tt):
    print("\nHold Out")
    x_train, x_test, y_train, y_test = pa.hold_out(X, Y, ts, tt)
    method_classification(x_train, x_test, y_train, y_test)
    
def method_random_subsampling(X, Y, n_splits, train , test, state):
    print("\nRandom Subsampling")
    x_train, x_test, y_train, y_test = pa.random_subsampling(X, Y, n_splits, train, test, state = state)
    method_classification(x_train, x_test, y_train, y_test)

def method_K_Fold(X,Y, n_splits, state, shuffle):
    print("\nK Fold: ")
    x_train, x_test, y_train, y_test = pa.K_fold(X, Y, n_splits, state, shuffle)
    method_classification(x_train, x_test, y_train, y_test)
    
def method_leave_one_out(X, Y):
    print("\nLeave One Out")
    print_header()
    Y_preds0, Y_preds1, Y_preds2, Y_preds3, Y_preds4 = pa.leave_one_out(X, Y)
    cm0 = method_metrics(Y, Y_preds0, "SGD      ")
    cm1 = method_metrics(Y, Y_preds1, "Trees    ")
    cm2 = method_metrics(Y, Y_preds2, "Vector M ")
    cm3 = method_metrics(Y, Y_preds3, "Neighbors")  
    cm4 = method_metrics(Y, Y_preds4, "NNModels ")
    plot_matriz_confusion(cm0, cm1, cm2, cm3, cm4)

def method_SCV(X, Y, n_splits):
    print("\nStratifield Cross Validation")
    x_train, x_test, y_train, y_test = pa.stratified_cross_validation(X, Y, n_splits)
    method_classification(x_train, x_test, y_train, y_test)
