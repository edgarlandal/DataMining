import warnings                                  # do not disturbe mode
warnings.filterwarnings('ignore')
import pandas as pd
import forestcasting as fc
import metrics as me
from tabulate import tabulate
import matplotlib.pylab as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

def plotting_datas(label ,y, y_train, y_test, yh):
    serie0 = pd.Series(y_train.ravel()) 
    serie1 = pd.Series(y_test.ravel(), 
                       index=(range(y_train.shape[0], y.shape[0]))) 
    serie2 = pd.Series(yh.ravel(), 
                       index=(range(y_train.shape[0], y.shape[0]))) 

    fig, ax = plt.subplots(figsize=(9, 4))
    plt.title(label)
    serie0.plot(ax=ax, label='train')
    serie2.plot(ax=ax, label='pred')
    serie1.plot(ax=ax, label='test')
    ax.legend();
    plt.show() 

def do_funtion(Y, X_train, Y_train, Y_test ,X_test, tipo, inputs, s_in):
    if (tipo == "Trees"):
        Y_pred = fc.decision_Tree(X_train, Y_train, X_test)
    elif (tipo == "Vector M"):
        Y_pred = fc.vector_machine(X_train, Y_train, X_test)
    elif (tipo == "RandomF"):
        Y_pred = fc.random_forest(X_train, Y_train, X_test, inputs, s_in)
    elif (tipo == "Linear"):
        Y_pred = fc.Linear(X_train, Y_train, X_test)
    
    plotting_datas(tipo,Y, Y_train, Y_test, Y_pred)
    r, r2, sse, mae, mse, rmse = me.method_metrics(Y_test, Y_pred, tipo)
    data = [tipo , r, r2, sse, mae, mse, rmse]
    return data

def method_regression(Y, X_train, X_test, Y_train, Y_test, inputs, s_in):
    # data0 = do_funtion(Y, X_train, Y_train, Y_test, X_test, "Trees")
    # data1 = do_funtion(Y, X_train, Y_train, Y_test, X_test, "Vector M")
    # data2 = do_funtion(Y, X_train, Y_train, Y_test, X_test, "Linear")
    data3 = do_funtion(Y, X_train, Y_train, Y_test, X_test, "RandomF", inputs, s_in)

    data = []
    # data.insert(0, data0)
    # data.insert(1, data1)
    # data.insert(2, data2)
    data.insert(0, data3)

    head = ["R", "R2", "SSE", "MAE", "MSE", "RMSE"]
    print(tabulate(data, headers=head, tablefmt="grid"))
    
