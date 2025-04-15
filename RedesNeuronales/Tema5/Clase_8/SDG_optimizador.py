

#SDG

class Optimizer_SDG:

    def __init__(self, learning_rate = 1.0):
        
        self.learning_rate = learning_rate
        pass

    def update_params(self, layer):
        #actualizar los pesos y los bias, - porque queremos ir hacia abajo
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
        pass