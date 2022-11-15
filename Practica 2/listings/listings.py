import pandas as pd
import seaborn as sns
import numpy as np
sns.set_palette('husl')

def missing_values(data):
    missing_values_count = data.isnull().sum()
    missing_values_count[0:40]
    
    total_cells = np.product(data.shape)
    total_missing = missing_values_count.sum()
    
    percent_missing = (total_missing/total_cells) * 100
    print(percent_missing)

def data_information(data):
    print(data.info())
    print(data.head())
    print(data.describe())
    
def delete_columns(data):
    return data.dropna(axis=1)

def remplace_for_constant(data):
    return data.fillna(method="bfill", axis=0).fillna(0)

def remplace_for_mean(data):
    dataAux = data
    for i in data:
        dataAux[i].fillna(data[i].mean, inplace = True)
    return dataAux

#1 - input data
data = pd.read_csv('listings.csv')
data_information(data)
missing_values(data)
data.hist()

#2 - Delete columns
data1 = delete_columns(data)
data_information(data1)
missing_values(data1)
data1.hist()

#3 - remplace for constant
data2 = remplace_for_constant(data)
data_information(data2)
missing_values(data2)
data2.hist()

#4 - remplace for mean
data3 = remplace_for_mean(data)
data_information(data3)
missing_values(data3)
data3.hist()