from sklearn.metrics import confusion_matrix

def matriz_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm

def accuracy(TP, TN, P , N):
    return (TP + TN)/(P + N)

def error_rate(FP, FN, P, N):
    return (FP + FN)/(P + N)

def presicion(TP, FP):
    if ((TP + FP) == 0):
        return 0
    return TP/(TP + FP)

def recall(TP, FN):
    if ((TP + FN) == 0):
        return 0
    return TP/(TP + FN)

def sensitivity(TP, P):
    return TP/P