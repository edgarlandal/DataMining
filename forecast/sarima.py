from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def SARIMA(data, train_size ,labels):    
    p = range(0, 4, 1)
    q = range(0, 4, 1)
    P = range(0, 4, 1)
    Q = range(0, 4, 1)
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    print(len(parameters_list))
    
    DIS = data.shape[0] - train_size
    best_model = SARIMAX(data, order=(0, 1, 2),
                         seasonal_order=(0, 1, 2, 4)).fit(dis=-DIS)
    
    best_model.plot_diagnostics(figsize=(15,12));
    
    data['arima_model'] = best_model.fittedvalues
    data['arima_model'][:train_size] = np.NaN 
        
    forecast = best_model.predict(start=data.shape[0] - 30, end=data.shape[0] + 200)
    forecast = data['arima_model'].append(forecast)
    
    print(forecast)
    
    plt.figure(figsize=(15, 7.5))
    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data[labels], label='actual')
    plt.legend()
    plt.show()
    