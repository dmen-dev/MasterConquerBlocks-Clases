import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


#cargar el archivo CSV
titanic_data = pd.read_csv("Titanic-Dataset.csv")


#Eliminar las columnas PassengerID, Name, Ticket, Cabin, Embarked
titanic_data_cleaned = titanic_data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

#Transformar los datos de la columna 'Sex' en binario: 0 para 'male' y 1 para 'female'
titanic_data_cleaned['Sex'] = titanic_data_cleaned['Sex'].map({'male':0, 'female':1})

#Mostrar las primeras filas del DataFrame resultante
#print(titanic_data_cleaned.head())

#quitar los valores incompletos
titanic_data_cleaned = titanic_data_cleaned.dropna()

#Extraigo resultado
Resultado = titanic_data_cleaned['Survived']
#print(len(Resultado))
Y = np.array(titanic_data_cleaned['Survived'])


#Elimino la columna del dataset para
titanic_data_cleaned = titanic_data_cleaned.drop(columns = ['Survived'])

#Construyo un array de numpy con los datos limpios
X = np.array(titanic_data_cleaned)


#ADAM
Capa_1_Adam = Capa_densa(6, 10)
Activacion_1_Adam = Activation_ReLu()
Capa_2_Adam = Capa_densa(10, 2)
loss_activation_Adam = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer_Adam = Optimizer_Adam(learning_rate = 0.05, decay = 1e-3)

#RMS
Capa_1_RMS = Capa_densa(6, 10)
Activacion_1_RMS = Activation_ReLu()
Capa_2_RMS = Capa_densa(10, 2)
loss_activation_RMS = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer_RMS = Optimizer_RMSProp(learning_rate = 0.05, decay = 1e-2)

#AdaGrad
Capa_1_AdaGrad = Capa_densa(6, 10)
Activacion_1_AdaGrad = Activation_ReLu()
Capa_2_AdaGrad = Capa_densa(10, 2)
loss_activation_AdaGrad = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer_AdaGrad = Optimizer_Adagrad(learning_rate = 0.05, decay = 1e-4)

Datos_Adam = {}
Datos_RMS = {}
Datos_AdaGrad = {}

for epoch in range(501):

    #Realizamos los FORWARDS

    #ADAM
    Capa_1_Adam.forward(X)
    Activacion_1_Adam.forward(Capa_1_Adam.output)
    Capa_2_Adam.forward(Activacion_1_Adam.output)
    loss_Adam = loss_activation_Adam.forward(Capa_2_Adam.output, Y)
    #RMS
    Capa_1_RMS.forward(X)
    Activacion_1_RMS.forward(Capa_1_RMS.output)
    Capa_2_RMS.forward(Activacion_1_RMS.output)
    loss_RMS = loss_activation_RMS.forward(Capa_2_RMS.output, Y)
    #AdaGrad
    Capa_1_AdaGrad.forward(X)
    Activacion_1_AdaGrad.forward(Capa_1_AdaGrad.output)
    Capa_2_AdaGrad.forward(Activacion_1_AdaGrad.output)
    loss_AdaGrad = loss_activation_AdaGrad.forward(Capa_2_AdaGrad.output, Y)


    #CALCULO DEL ERROR
    #ADAM
    predicciones = np.argmax(loss_activation_Adam.output, axis = 1)

    if len(Y.shape) == 2:
        Y = np.argmax(Y, axis = 1)
    precision_Adam = np.mean(predicciones == Y)

    #RMS
    predicciones = np.argmax(loss_activation_RMS.output, axis = 1)

    if len(Y.shape) == 2:
        Y = np.argmax(Y, axis = 1)
    precision_RMS = np.mean(predicciones == Y)

    #AdaGrad
    predicciones = np.argmax(loss_activation_AdaGrad.output, axis = 1)

    if len(Y.shape) == 2:
        Y = np.argmax(Y, axis = 1)
    precision_AdaGrad = np.mean(predicciones == Y)

    #GUARDO LOS DATOS
    if not epoch % 5: 
        Datos_Adam[epoch] = {
            'Precision': precision_Adam,
            'Pérdida': loss_Adam,
            'Learning_rate': optimizer_Adam.current_learning_rate
        }
        Datos_RMS[epoch] = {
            'Precision': precision_RMS,
            'Pérdida': loss_RMS,
            'Learning_rate': optimizer_RMS.current_learning_rate
        }
        Datos_AdaGrad[epoch] = {
            'Precision': precision_AdaGrad,
            'Pérdida': loss_AdaGrad,
            'Learning_rate': optimizer_AdaGrad.current_learning_rate
        }

    #BACKWARDS
    
    #ADAM
    loss_activation_Adam.backwards(loss_activation_Adam.output, Y)
    Capa_2_Adam.backwards(loss_activation_Adam.dinputs)
    Activacion_1_Adam.backwards(Capa_2_Adam.dinputs)
    Capa_1_Adam.backwards(Activacion_1_Adam.dinputs)
    
    optimizer_Adam.pre_update_params()
    optimizer_Adam.update_params(Capa_1_Adam)
    optimizer_Adam.update_params(Capa_2_Adam)
    optimizer_Adam.post_update_params()

    
    #RMS
    loss_activation_RMS.backwards(loss_activation_RMS.output, Y)
    Capa_2_RMS.backwards(loss_activation_RMS.dinputs)
    Activacion_1_RMS.backwards(Capa_2_RMS.dinputs)
    Capa_1_RMS.backwards(Activacion_1_RMS.dinputs)

    optimizer_RMS.pre_update_params()
    optimizer_RMS.update_params(Capa_1_RMS)
    optimizer_RMS.update_params(Capa_2_RMS)
    optimizer_RMS.post_update_params()

    #AdaGrad
    loss_activation_AdaGrad.backwards(loss_activation_AdaGrad.output, Y)
    Capa_2_AdaGrad.backwards(loss_activation_AdaGrad.dinputs)
    Activacion_1_AdaGrad.backwards(Capa_2_AdaGrad.dinputs)
    Capa_1_AdaGrad.backwards(Activacion_1_AdaGrad.dinputs)

    optimizer_AdaGrad.pre_update_params()
    optimizer_AdaGrad.update_params(Capa_1_AdaGrad)
    optimizer_AdaGrad.update_params(Capa_2_AdaGrad)
    optimizer_AdaGrad.post_update_params()    

#Convierto el diccionario en un Dataframe

df_Adam = pd.DataFrame.from_dict(Datos_Adam, orient='index')
df_RMS = pd.DataFrame.from_dict(Datos_RMS, orient='index')
df_AdaGrad = pd.DataFrame.from_dict(Datos_AdaGrad, orient='index')

#Extraigo las metricas en una variable 
epoch = df_Adam.index

#ADAM
precision_Adam = df_Adam['Precision']
perdida_Adam = df_Adam['Pérdida']
learning_rate_Adam = df_Adam['Learning_rate']

#RMS
precision_RMS = df_RMS['Precision']
perdida_RMS = df_RMS['Pérdida']
learning_rate_RMS = df_RMS['Learning_rate']

#AdaGrad
precision_AdaGrad = df_AdaGrad['Precision']
perdida_AdaGrad = df_AdaGrad['Pérdida'] 
learning_rate_AdaGrad = df_AdaGrad['Learning_rate']

#Creamos el gráfico
fig, axs = plt.subplots(3, 1, figsize=(6,9))
tamano = 20

#Grafico de precision
axs[0].set_title('Evolución frente a las iteraciones')
axs[0].set_ylabel('Precision')
axs[0].scatter(epoch, precision_Adam, marker = '.', color = 'r', label = 'ADAM', s = tamano )
axs[0].scatter(epoch, precision_RMS, marker = '.', color = 'b', label = 'RMS', s = tamano )
axs[0].scatter(epoch, precision_AdaGrad, marker = '.', color = 'g', label = 'AdaGrad', s = tamano )

axs[0]. grid(True)

#Grafico de pérdida
axs[1].set_ylabel('Perdida')
axs[1].scatter(epoch, perdida_Adam, marker = '.', color = 'r', label = 'ADAM', s = tamano )
axs[1].scatter(epoch, perdida_RMS, marker = '.', color = 'b', label = 'RMS', s = tamano )
axs[1].scatter(epoch, perdida_AdaGrad, marker = '.', color = 'g', label = 'AdaGrad', s = tamano )

axs[1]. grid(True)

#Grafico de pérdida
axs[2].set_ylabel('Learning Rate')
axs[2].scatter(epoch, learning_rate_Adam, marker = '.', color = 'r', label = 'ADAM', s = tamano )
axs[2].scatter(epoch, learning_rate_RMS, marker = '.', color = 'b', label = 'RMS', s = tamano )
axs[2].scatter(epoch, learning_rate_AdaGrad, marker = '.', color = 'g', label = 'AdaGrad', s = tamano )

axs[2]. grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()
