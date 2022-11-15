import math
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error

def R(y, yh):
    if(y.shape[1] == 1):
        r = np.corrcoef(y.T, yh.T)
    else:
        r = np.corrcoef(y.T[0], yh.T[0])
    return r[0][1]

def R2(y, yh):
    return r2_score(y, yh)

def SSE(y, yh):
  return np.sum((y - yh)**2)

def MAE(y, yh):
    return median_absolute_error(y, yh)

def MSE(y, yh):
  return mean_squared_error(y, yh)

def RMSE(y, yh):
  return math.sqrt(MSE(y, yh))

def method_metrics(y, yh, tipo):
    r = R(y, yh)
    r2 = R2(y, yh)
    sse = SSE(y, yh)
    mae = MAE(y, yh)
    mse = MSE(y, yh)
    rmse = RMSE(y, yh)
    return r, r2, sse, mae, mse, rmse
