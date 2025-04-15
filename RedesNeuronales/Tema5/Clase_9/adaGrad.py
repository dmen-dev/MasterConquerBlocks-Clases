
import numpy as np

class Optimizer_Adagrad:

    def __init__ (self, learning_rate =1., decay=0., epsilon=1e-7): #cuando los números son muy pequeños, se puede dar el caso de que no se pueda dividir entre 0
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        pass

    def pre_update_params(self):
        
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1+self.decay * self.iteration))

    def update_params(self, layer):

        #si no existe el cache lo creamos
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #calculamos el acumulado del cache
        layer.weight_cache += layer.dweights ** 2 #conseguimos acumular el cuadrado de los gradientes para cada uno de los pesos
        layer.bias_cache += layer.dbiases ** 2

        #actualiza la tasa de aprendizaje en base a la raíz cuadrada de los históricos de los pesos y sesgos
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)    

    def post_update_params(self):

        self.iteration += 1

