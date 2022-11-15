import warnings                                  # do not disturbe mode
warnings.filterwarnings('ignore')
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

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
       
def Linear(X, Y, X_test):
    if(Y.shape[1] == 1):
        reg = LinearRegression()
    else:
        reg = MultiOutputRegressor(LinearRegression())
    return reg.fit(X, Y).predict(X_test)     

def random_forest(X, Y, X_test, inputs, s_in):
    if(Y.shape[1] == 1):
        reg = RandomForestRegressor()
    else:
        reg = MultiOutputRegressor(RandomForestRegressor())
    
    reg = reg.fit(X, Y)
    
    return forecast(X, X_test, reg, inputs, s_in)


def forecast(X, X_test, reg, inputs, s_in):
    X_aux = X.flatten()             #vectorizar
    max = inputs + s_in             #tope de los datos con saltos
    data = X_aux[-max:]             #tomar solo los datos finales
    
    Y = []                          #datos auxtiliares
    X_a = []
    data = np.array(data)
    for i in range(X_test.shape[0]):#Datos a predecir
        j = 0       
        while(j < max):             #Llenar para predecir dato
            if(j == 0):
                X_a.append(data[j]) #se almacena para predecir
            else:
                j = j + s_in        #reposiciona
                X_a.append(data[j]) #se almacena para predecir
            j +=1        
        y = reg.predict([X_a])
        Y.append(y)  #se predice y lo guarda
        X_a = []
        
        data = np.append(arr = data[-(max-1):], values = y, axis = 0) #mueve ventana
    return np.array(Y)