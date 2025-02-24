import numpy as np
import matplotlib.pyplot as plt

class Capa_densa:
    #incio de la capa
    def __init__(self, n_inputs, n_neuronas):
        #inicio los pesos de la capa con aleatorio
        self.weights = 0.01 * np.random.rand(n_inputs, n_neuronas)
        #inicio de los sesgos de la capa
        self.biases = np.zeros((1, n_neuronas))
        pass

    def forward(self, inputs):
        #calcula los output de la capa a través del producto escalar
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

A = Capa_densa(2,5)

x = np.linspace(0,2*np.pi, num=100)
y = np.cos(x)

#plt.scatter(x,y)
#plt.show()

#Creamos un array con los inputs
x = np.array(x)
y = np.array(y)

#Creamos una matriz con nuestros datos
Data = np.vstack((x,y)).T
#print(Data)

#Creamos la capa 1
Capa_1 = Capa_densa(2,3)

#Ejecutamos las salidas de la capa
Capa_1.forward(Data)

#Salida de la capa
print(f"Salida de la Capa 1: {Capa_1.output}")
#print(f"Salida de la Capa 1: {Capa_1.output[:5]}")

Capa_2 = Capa_densa(3,6)

Capa_2.forward(Capa_1.output)


print(f"Salida de la Caap 2: {Capa_2.output[:5]}")