import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

data = pd.read_csv('iris.csv')

data.head()
data.info()
print(data.describe(include='all'))

data['species'].value_counts()
tmp = data.drop('id', axis=1)
#Pairplot
g = sns.pairplot(tmp, hue='species', markers='+')
plt.show()

#histograma de iris
tmp.hist()

#plotbox de iris
tmp.plot.box(by = 'species',fontsize = 8,figsize=(10, 8) ,rot = 90) 

#Scatterplot
plt.figure(figsize = (12,6))
plt.title('Petal Dimensions')
sns.scatterplot(x="petal_length", y="petal_width", hue='species', data = data);

plt.figure(figsize = (12,6))
plt.title('Sepal Dimensions')
sns.scatterplot(x="sepal_length", y="sepal_width", hue='species', data = data);