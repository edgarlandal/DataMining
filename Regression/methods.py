import pull_apart as pa
import regression as reg
import metrics as me
from tabulate import tabulate

def do_funtion(X_train, Y_train, Y_test ,X_test, tipo):
    if (tipo == "Trees"):
        Y_pred = reg.decision_Tree(X_train, Y_train, X_test)
    elif (tipo == "Vector M"):
        Y_pred = reg.vector_machine(X_train, Y_train, X_test)
    elif (tipo == "Logistic"):
        Y_pred = reg.Logistic(X_train, Y_train, X_test)
        
    r, r2, sse, mae, mse, rmse = me.method_metrics(Y_test, Y_pred, tipo)
    data = [tipo , r, r2, sse, mae, mse, rmse]
    return data

def method_regression(X_train, X_test, Y_train, Y_test):
    data0 = do_funtion(X_train, Y_train, Y_test, X_test, "Trees")
    data1 = do_funtion(X_train, Y_train, Y_test, X_test, "Vector M")
    data2 = do_funtion(X_train, Y_train, Y_test, X_test, "Logistic")
    
    data = []
    data.insert(0, data0)
    data.insert(1, data1)
    data.insert(2, data2)

    head = ["R", "R2", "SSE", "MAE", "MSE", "RMSE"]
    print(tabulate(data, headers=head, tablefmt="grid"))
    
def method_hold_out(X, Y, ts, tt):
    print("\nHold Out")
    x_train, x_test, y_train, y_test = pa.hold_out(X, Y, ts, tt)
    method_regression(x_train, x_test, y_train, y_test)
    
def method_random_subsampling(X, Y, n_splits, train , test, state):
    print("\nRandom Subsampling")
    x_train, x_test, y_train, y_test = pa.random_subsampling(X, Y, n_splits, train, test, state = state)
    method_regression(x_train, x_test, y_train, y_test)

def method_K_Fold(X,Y, n_splits, state, shuffle):
    print("\nK Fold: ")
    x_train, x_test, y_train, y_test = pa.K_fold(X, Y, n_splits, state, shuffle)
    method_regression(x_train, x_test, y_train, y_test)