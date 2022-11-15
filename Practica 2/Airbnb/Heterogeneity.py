import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

def data_information(data):
    print(data.info())
    print(data.describe())
    print(data.head())
    print(data.isnull().sum())
    data.hist()

#1
data = pd.read_csv('Phones_accelerometer.csv')
data_information(data)

#2
data = pd.read_csv('Phones_gyroscope.csv')
data_information(data)

#3
data = pd.read_csv('Watch_accelerometer.csv')
data_information(data)

#4
data = pd.read_csv('Watch_gyroscope.csv')
data_information(data)

                                                                                                                                                                                                                                                                                                                      