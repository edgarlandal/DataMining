import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mad_funtion(data):
    result = data.mad()
    print(result)
    result.plot(kind = 'bar')
    plt.show()
    
data = pd.read_csv('iris.csv')
mad_funtion(data = data)

data2 = pd.read_csv('2015.csv')
data2 = data2.drop('Happiness Rank', axis=1)
mad_funtion(data = data2)