import pandas as pd
import procesing_data as proda
import methods as me
import sarima as s
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

""" Dataset 1 """
data = pd.read_csv('data1.csv')

# Load/split your data

# s.SARIMA(data, 768, '1.2')

""" Dataset 1 : Ventana 1"""
dataset = proda.create_window(data,2,1,2,0)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 2)

""" Dataset 1 : Ventana 2"""
dataset = proda.create_window(data,2,1,4,0)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 4)

""" Dataset 1 : Ventana 3"""
dataset = proda.create_window(data,2,1,0,1)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 0)

""" Dataset 1 : Ventana 3"""                           
dataset = proda.create_window(data,2,1,0,2)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 0)

""" Dataset 2 """

data = pd.read_csv('data2.csv')
data = data.iloc[:,1]
data = data.iloc[:1000]

data = data.to_numpy().reshape(1000,1)
scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns = ['name'])

# s.SARIMA(data, 600, 'name')

""" Dataset 2 : Ventana 1"""
dataset = proda.create_window(data,2,1,0,0)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 0)

""" Dataset 2 : Ventana 2"""
dataset = proda.create_window(data,2,1,4,0)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 4)

""" Dataset 2 : Ventana 3"""
dataset = proda.create_window(data,2,1,0,1)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 0)

""" Dataset 2 : Ventana 3"""                           
dataset = proda.create_window(data,2,1,0,2)
X, y = proda.separator(dataset,1)
x_train, x_test ,y_train, y_test = proda.separate_data(X, y, 0.6)
me.method_regression(y, x_train, x_test, y_train, y_test, 2, 0)
