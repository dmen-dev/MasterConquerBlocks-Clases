import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

#Simular el proceso de dropout

num_neuronas = 100
lambda_poisson = 0.5
num_interaciones = 1000

#Simular el proceso de apagado de neuronas
def simular_dropout(num_neuronas,lambda_poisson):
    
    return poisson.rvs(mu = lambda_poisson, size = num_neuronas)

num_neuronas_off_counts = [] #cantidad de neuronas apagadas en cada iteracion

for _ in range(num_interaciones):
    dropout_mask = simular_dropout (num_neuronas, lambda_poisson)
    #print(dropout_mask)
    num_neuronas_off = np.sum (dropout_mask)
    num_neuronas_off_counts.append(num_neuronas_off)

#Calcular la media de neuronas off
mean_neuronas_off = np.mean(num_neuronas_off_counts)
print(f'Numero medio de neuronas desactivadas en {num_interaciones} iteraciones son {mean_neuronas_off:.2f}')

#Hago un histograma y muestro numero medio de neuronas desactivadas
plt.hist(num_neuronas_off_counts, bins=20, density=True,alpha=0.6,color ='g', edgecolor= 'black')
plt.title('Distribución de neuronas desactivadas en 1000 iteraciones')
plt.xlabel('Numero de neuronas desactivadas')
plt.ylabel('Frecuencia')
plt.show()