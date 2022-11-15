from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

def decision_Tree(X, Y, X_test):
    if(Y.shape[1] == 1):    
        reg = tree.DecisionTreeRegressor()
    else:
        reg = MultiOutputRegressor(tree.DecisionTreeRegressor())
        
    return reg.fit(X, Y).predict(X_test)     

def vector_machine(X, Y, X_test):
    if(Y.shape == 1):
        reg =  svm.SVR()
    else:
        reg =  MultiOutputRegressor(svm.SVR())
        
    return reg.fit(X, Y).predict(X_test)
       
def Logistic(X, Y, X_test):
    if(Y.shape[1] == 1):
        reg = LinearRegression()
    else:
        reg = MultiOutputRegressor(LinearRegression())
    return reg.fit(X, Y).predict(X_test)     

