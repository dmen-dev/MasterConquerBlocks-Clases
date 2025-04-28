import pandas as pd
import numpy as np
import json

class Capa_densa:
    #incio de la capa
    def __init__(self, n_inputs, n_neuronas):
        #inicio los pesos de la capa con aleatorio
        self.weights = 0.01 * np.random.randn(n_inputs, n_neuronas)
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

            #Gradiente de los valores
            self.dinputs = np.dot(dvalues, self.weights.T)

            pass

class Activation_ReLu:

    def forward(self, inputs):
        self.inputs = inputs
        #calculamos el output

        self.output = np.maximum(0, inputs)

    def backwards(self, dvalues):
        #como tenemos que modificar los valores haremos ujna copia de los mismos
        self.dinputs = dvalues.copy()
        #el gradiente es 0 cuando los valores son negativos porque la neurona esta desactivada
        self.dinputs[self.inputs <= 0] = 0

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

class Loss:
    def calculate(self, output, y):
        # CALCULAR LA PÉRDIDA DE MUESTRA
        sample_losses = self.forward(output, y)
        # CALCULAR LA PÉRDIDA MEDIA
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # CALCULO DEL NUMERO DE DATOS
        n_datos = len(y_pred)

        # LIMITAR LOS VALORES DE Y_PRED PARA EVITAR LOG(0)
        y_pred_limite = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # SOLO PARA VALORES CATEGORICOS
        if len(y_true.shape) == 1:  # ES IGUAL A UN VECTOR
            confianza = y_pred_limite[range(n_datos), y_true]
        elif len(y_true.shape) == 2:  # ES IGUAL A UNA MATRIZ
            confianza = np.sum(y_pred_limite * y_true, axis=1)

        # CALCULAR LA PÉRDIDA NEGATIVA
        Perdida_negativa = -np.log(confianza)

        return Perdida_negativa
    
class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, y_true):

        #Funcion de activación de la capa de salida
        self.activation.forward(inputs)
        
        #fijamos el output
        self.output = self.activation.output

        #calculamos y devolvemos la pérdida
        return self.loss.calculate(self.output, y_true)
    
    def backwards(self, dvalues, y_true):

        #numero de muestras, lo utilizaremos para normalizar los valores
        muestras = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1) #argmax calcula el máximo de cada una de las filas, axis = 1 para hacerlo a lo largo de las columnas

        #copiamos los valores para modificarlos sin cambiar los previos
        self.dinputs = dvalues.copy()

        #calculamos el gradiente
        #se pone -1 para calcularlo para cada una de las muestras
        self.dinputs[range(muestras), y_true] -= 1

        #normalizamos el gradiente
        self.dinputs = self.dinputs / muestras

class Optimizer_SDG:

    #Iniciamos el optimizador fijando una tasa de aprendizaje o learning_rate = 1
    # Esta tasa es la que se utiliza para actualizar los parámetros

    def __init__(self, learning_rate = 1.0, decay = 0.0, momentum = 0.0):
        
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
        #actualizamos los parámetros

    def pre_update_params(self):

        if self.decay:
            self.current_learnign_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer):

        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                #creo el array para los pesos
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.biases_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate * layer.dbiases
            layer.biases_momentums = biases_updates

        else:
            weight_updates += -self.current_learning_rate * layer.dweights
            biases_updates += -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += biases_updates

    def post_update_params(self):
        self.iterations += 1

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

class Optimizer_RMSProp:

    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.rho = rho

        pass

    def pre_update_params(self):

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iteration))

    def update_params(self, layer):
        
        # Si no existe el cache lo creamos
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Actualizo los caches en función de la media móvil exponencial
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # Actualizo los pesos y sesgos
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
            
        self.iteration += 1

class Optimizer_Adam:

    def __init__ (self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999): #beta1 significa que decaymiento (Descenso) es lento, si fuera un valor pequeño hace que pierda la inercia que llevaba
    #beta_2 controla el descenso más lento si es un valor pequeño, si es un valor grande hace que el descenso sea más rápido. Hace que sea dificil caer en mínimos locales            

        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2        

    def pre_update_params (self):

        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1.+self.decay * self.iteration))

    def update_params (self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Actualizo los caches de los gradientes actuales
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        #Obtengo el momento corregido
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 **(self.iteration + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iteration + 1))

        #Actualizo el cache corregido
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1- self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        #Obtengo el cache corregido
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iteration +1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iteration + 1))

        #Actualizo los pesos y sesgos
        layer.weights += -self.learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += -self.learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_update_params (self):

        self.iteration += 1


def cargar_datos_json(ruta_json):
    with open(ruta_json, 'r') as archivo_json:
        model_parameters = json.load(archivo_json)

    capa1_datos = model_parameters['Capa1']
    capa2_datos = model_parameters['Capa2']

    export = {
        'Capa1.weights': np.array(capa1_datos['Weights']),
        'Capa1.biases': np.array(capa1_datos['Biases']),
        'Capa2.weights': np.array(capa2_datos['Weights']),
        'Capa2.biases': np.array(capa2_datos['Biases'])
    }

    return export



#los inputs son 6 neuronas, porque son 6 las columnas que tenemos en el dataset y 10 neuronas de salida (las de salida nos inventamos)
#Creamos primera capa oculta
Capa_1 = Capa_densa(6, 10)

#Creamos funcion de activación
Activacion_1 = Activation_ReLu()

#Creamos segunda capa oculta
#Se ponen 2 output ya que necesitamos saber si vive o muere
Capa_2 = Capa_densa(10, 2)
#Activacion_2 = Activation_softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

parametros = cargar_datos_json('D:\VSCode\MasterConquerBlocks\RedesNeuronales\Tema5\Clase_13\model_parameters.json')

#Fijo los parámetros de peso
Capa_1.weights = parametros['Capa1.weights']
Capa_2.weights = parametros['Capa2.weights']

#Fijo los parámetros de sesgo
Capa_1.biases = parametros['Capa1.biases']
Capa_2.biases = parametros['Capa2.biases']

Yo = {
    'Pclass': 1,
    'Sex': 1,
    'Age': 30,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 500,
}

X = np.array(list(Yo.values()))

#print(X)
Capa_1.forward(X)
Activacion_1.forward(Capa_1.output)
Capa_2.forward(Activacion_1.output)
loss = loss_activation.forward(Capa_2.output, np.array([0])) 

#print(loss_activation.output)
#0 --> [1, 0] NO SUPERVIVENCIA
#1 --> [0, 1] SUPERVIVENCIA

print('Posibilidad de supervivencia: ', round(loss_activation.output[0][1]*100, 2), '%')