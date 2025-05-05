import pandas as pd
import numpy as np

class Capa_densa:
    #incio de la capa
    def __init__(self, n_inputs, n_neuronas,weight_regularizer_l1 = 0, weight_regularizer_l2 = 0, bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):
        #inicio los pesos de la capa con aleatorio
        self.weights = 0.01 * np.random.rand(n_inputs, n_neuronas)
        #inicio de los sesgos de la capa
        self.biases = np.zeros((1, n_neuronas))
        pass

    def forward(self, inputs):
        #calcula los output de la capa a través del producto escalar
        self.output = np.dot(inputs, self.weights) + self.biases
        
        self.inputs = inputs #necesitamos esto para derivar la función, para recordar los inputs

        pass

    def backwards(self, dvalues):

            #Gradiente de los parámetros

            self.dweights = np.dot(self.inputs.T, dvalues) #se traspone para que se pueda multiplicar
            self.dbiases = np.sum(dvalues, axis = 0, keepdims= True) 

            #Regularización de los pesos de L1
            if self.weight_regularizer_l1 > 0:
                dL1 = np.ones_like(self.weights)
                dL1[self.weights < 0] = -1

                self.dweights += self.weights_regularizer_l1 * dL1
            
            #Regularización de los pesos de L2
            if self.weights_regularizer_l2 > 0:
                self.dweights += 2 * self.weights_regularizer_l2 * self.weights

            #L1 en los sesgos
            if self.bias_regularizer_l1 > 0:
                dL1 = np.ones_like(self.biases)
                dL1[self.biases < 0] = -1

                self.dbiases += self.bias_regularizer_l1 * dL1

            #L2 en los sesgos
            if self.bias_regularizer_l2 > 0:
                self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

            #Gradiente de los valores
            self.dinputs = np.dot(dvalues, self.weights.T)

            