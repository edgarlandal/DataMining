import pandas as pd
import methods as me
import numpy as np

#preprossesing  
def separator(data, output, delete):
    inputs = data.drop(labels=delete, axis= 1)
    targets = data[output]
    return inputs, targets

def delete_rows(data):
    return data.dropna(axis=0)

def delete_colums(data):
    return data.dropna(axis=1)

def df_to_np(X, Y):
    return X.to_numpy(), Y.to_numpy()

def to_float(data):
    for i in data:
        data[i] = data[i].astype(float)
    return data
    
def delete_colums_small(data):
    delete = []
    for i in data:
        if(data[i].count() < 2000):
            delete.append(i)
    return data.drop(labels=delete, axis= 1)

def preprossessiong_data3(data):
   return to_float(delete_rows(delete_colums_small(remplace_sig(data))))
    
def remplace_sig(data):
    for i in data:
        data[i] = data[i].replace('?', np.NaN)
    return data

def get_output(data, start, stop):
    output = []
    for i in range(start, stop,1):
        output.append(data.columns[i])
    return output

def start_model(X, Y):
    X, Y = df_to_np(X, Y)
    me.method_hold_out(X, Y, 0.6, 0.4)
    me.method_random_subsampling(X, Y, 30, 0.5 , 0.25, 0)
    me.method_K_Fold(X, Y, 5, False, True)
    me.method_K_Fold(X, Y, 10, False, True)

""" Dataset 1 """

data = pd.read_csv('data1.csv')
print(data.info())
X, Y = separator(data, output=['price'], delete=['price'])
start_model(X, Y)

""" Dataset 2 """

data = pd.read_csv('data2.csv')
data = delete_rows(data)
print(data.info())
X, Y = separator(data, output=['Next_Tmax','Next_Tmin'], delete=['Next_Tmax','Next_Tmin', 'Date'])
start_model(X, Y)

""" Dataset 3 """
data = pd.read_csv('data3.csv') 
data = preprossessiong_data3(data)
print(data.info())  
output = get_output(data,25, 38)
X, Y = separator(data, output= output, delete = output)
start_model(X, Y)