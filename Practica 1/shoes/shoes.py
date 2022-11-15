# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 17:38:30 2022

@author: mikeydrako123
"""
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

data = pd.read_csv('MEN_SHOES.csv')

data.head()
data.info()
print(data.describe(include='all'))

data.isnull().sum()
data = data.drop('Product_details', axis=1)

data['Current_Price'].unique()

data['How_Many_Sold'] = data['How_Many_Sold'].str.replace(',', '')
data['Current_Price'] = data['Current_Price'].str.replace(',', '').str.replace('₹', '')
data['Current_Price'].unique()
data.dropna(axis = 0, inplace = True)
data['Current_Price'] = data['Current_Price'].astype(int)
data['How_Many_Sold'] = data['How_Many_Sold'].str.replace(',', '')
data['How_Many_Sold'] = data['How_Many_Sold'].astype(int)


#Boxplot
sns.boxplot(x='Brand_Name', y='Current_Price', data=data)
plt.xticks(rotation=90)
plt.xlabel("Shoes Sold",fontsize=12)
plt.ylabel("Price",fontsize=12)
plt.title("Shoes Sold vs Price Scatter",fontsize=20)
plt.show()

#Pairplot
h = data['Brand_Name'].value_counts()
g = sns.pairplot(data,diag_kind='kde')
plt.show()

#Histograma
data.hist()

#Scatterplot
plt.figure(figsize = (12,6))
sns.scatterplot(x="How_Many_Sold", y="Current_Price", hue='Brand_Name', data = data);
