import seaborn as sns
import matplotlib.pyplot as plt

datos = sns.load_dataset('iris')

#print(data)

#Grafico de dispersion
sns.scatterplot(data=datos,x='sepal_length', y='sepal_width')
plt.title('Gráfico de dispersión de longitud de sepalo vs anchura')
plt.xlabel = 'sepal_length'
plt.ylabel = 'sepal_width'
#plt.show()

#Grafico de barras
sns.barplot(data=datos, x='species', y='sepal_length')
plt.title('Gráfico de barras de longitud de sepalo vs especie')
plt.xlabel = 'species'
plt.ylabel = 'sepal_length'
#plt.show()

#Histograma
sns.histplot(datos['sepal_length'],bins=5)
plt.title('Histograma de longitud de sepalo')
plt.xlabel = 'sepal_length'
plt.ylabel = 'numero_repeticiones'
#plt.show()

#Gráfico de violín
sns.violinplot(data = datos, x = 'species', y = 'sepal_length')
plt.title('Gráfico de violín')
#plt.show()


sns.pairplot(data = datos, hue = 'species')
plt.show()
#heatmap
corr = datos.corr()
sns.heatmap(corr, annot=True,cmap='coolwarm')
plt.title('Mapa de calor de las correlaciones')
plt.show()