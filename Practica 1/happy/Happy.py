import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

data = pd.read_csv('2015.csv')

data.head()
data.info()
print(data.describe(include='all'))

#Pairplot
g = sns.pairplot(data, hue='Region')
g = sns.pairplot(data, hue='Country')

#Boxplot
sns.boxplot(x='Region', y='Happiness Score', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Standard Error', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Health (Life Expectancy)', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Family', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Economy (GDP per Capita)', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Freedom', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Generosity', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

sns.boxplot(x='Region', y='Dystopia Residual', data=data)
plt.xticks(rotation=90)
plt.figure(figsize = (12,12))
plt.show()

#Histograma
data.hist()

#Scatterplot
sns.set_theme(style='dark')
plt.figure(figsize=(24,12))
ax=sns.scatterplot(data=data,x='Country',y='Health (Life Expectancy)',size='Economy (GDP per Capita)',sizes=(500,2000)
                   ,hue='Economy (GDP per Capita)')
plt.xticks(rotation=90,fontsize=15,color='midnightblue')
plt.yticks(fontsize=15,color='midnightblue')
plt.xlabel('Country',size=15,color='midnightblue')
plt.ylabel('Healthy life expectancy',size=15,color='midnightblue')
plt.show()

sns.set_theme(style='dark')
plt.figure(figsize=(24,12))
ax=sns.scatterplot(data=data,x='Region',y='Health (Life Expectancy)',size='Economy (GDP per Capita)',sizes=(500,2000)
                   ,hue='Economy (GDP per Capita)')
plt.xticks(rotation=90,fontsize=12,color='midnightblue')
plt.yticks(fontsize=15,color='midnightblue')
plt.xlabel('Country',size=15,color='midnightblue')
plt.ylabel('Healthy life expectancy',size=15,color='midnightblue')
plt.show()