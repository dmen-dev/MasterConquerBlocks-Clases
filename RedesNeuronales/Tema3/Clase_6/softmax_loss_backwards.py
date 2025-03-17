import numpy as np

class Activation_softmax:

    def forward(self, inputs):
        #NUËVO recuerda los inputs para la optimización
        self.inputs = inputs
        #calcula las probabilidades no normalizadas
        exp_values = np.exp(inputs-np.max(inputs, axis = 1, keepdims = True))

        #normalizo las probabilidades
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

        self.output = probabilities

    def backwards(self, dvalues):

        #Crear un array vacío
        self.dinputs = np.empty_like(dvalues) #crea un array vacío con las mismas dimensiones que dvalues

        #generar cada uno de los gradientes para cada uno de las dimensiones de la matriz
        for index,(single_output, single_values) in enumerate(zip(self.output, dvalues)): #recorre el array de output y dvalues. 
            #La función zip crea un array donde cruza los valores como una cremallera
            
            #para operar con la fila, debemos "aplanarlo"
            single_output = single_output.reshape(-1, 1)
            
            #calculamos la matriz jacobiana de la función output
            #diagflat convierte un array en una matriz diagonal
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            #calcular el gradiente del output
            #np.dot realiza el producto escalar que significa: multiplicar los valores de la matriz
            self.dinputs[index] = np.dot(jacobian_matrix, single_values)