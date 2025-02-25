import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 40)
y = np.exp(x)

#plt.plot(x,y)
#plt.show()

outputs = [3, 5, -3]

exp_valores = np.exp(outputs)

suma_exponenciales = sum(exp_valores)

normalizados = exp_valores / suma_exponenciales

#print(normalizados)

class Activation_softmax:

    def forward(self, inputs):
        #calculo los valores exponenciales restandoles el valor maximo
        exp_values = np.exp(inputs-np.max(inputs, keepdims = True))

        #normalizo las probabilidades
        probabilidades = exp_values / np.sum(exp_values, keepdims = True)

        self.output = probabilidades

outputs = [1,2,3]

AS = Activation_softmax()
AS.forward(outputs)
print(AS.output)