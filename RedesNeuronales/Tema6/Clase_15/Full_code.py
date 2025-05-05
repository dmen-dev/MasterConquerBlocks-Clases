import pandas as pd
import numpy as np

class Capa_densa:
    #incio de la capa
    def __init__(self, n_inputs, n_neuronas, weight_regularizer_l1 = 0, weight_regularizer_l2 = 0, bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):
        #inicio los pesos de la capa con aleatorio
        self.weights = 0.01 * np.random.rand(n_inputs, n_neuronas)
        #inicio de los sesgos de la capa
        self.biases = np.zeros((1, n_neuronas))
        
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

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

                self.dweights += self.weight_regularizer_l1 * dL1
            
            #Regularización de los pesos de L2
            if self.weight_regularizer_l2 > 0:
                self.dweights += 2 * self.weight_regularizer_l2 * self.weights

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
    
    def  regularization_loss(self, layer):
        #se le aplicará a cada una de las capas
        regularization_loss = 0
        #Regularización de los pesos de L1
        
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

        #Regularización de los pesos de L2

        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss

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

class Layer_Dropout:

    def __init__(self, dropout_rate):
        self.dropout_rate = 1 - dropout_rate
        pass

    def forward(self, inputs):

        self.inputs = inputs #fijo los inputs

        #mask binomial y reescalado
        self.binary_mask = np.random.binomial(1, self.dropout_rate, size = inputs.shape) / self.dropout_rate #se divide por rate para mantener valores +- constantes

        self.output = inputs * self.binary_mask

    def backwards (self, dvalues):

         self.dinputs = dvalues * self.binary_mask   

#cargar el archivo CSV
#file_path = '/Clase_3/Titanic-Dataset.csv'
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
#print(X)

#quitar los valores no completaos, aquellos  como nan


#los inputs son 6 neuronas, porque son 6 las columnas que tenemos en el dataset y 10 neuronas de salida (las de salida nos inventamos)
#Creamos primera capa oculta
#Capa_1 = Capa_densa(6, 10)
Capa_1 = Capa_densa(6, 100, weight_regularizer_l2 = 5e-4, bias_regularizer_l2 = 5e-4)

#Creamos funcion de activación
Activacion_1 = Activation_ReLu()

dropout_1 = Layer_Dropout(0.5)

#Creamos segunda capa oculta
#Se ponen 2 output ya que necesitamos saber si vive o muere
Capa_2 = Capa_densa(100, 2)
#Activacion_2 = Activation_softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

optimizer = Optimizer_Adam(decay = 1e-4)
#optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 1e-4)
#optimizer = Optimizer_Adam(leraning_rate = 0.01, decay = 1e-6)




for epoch in range(501):
    #Iniciamos la red con valores del dataframe. Como entrada todos los datos del titanic habiendo eliminado la columna de supervivientes
    Capa_1.forward(X)

    #Funcion de activacion sobre output de capa 1
    Activacion_1.forward(Capa_1.output)

    #Tras la función de activación se aplica el dropout
    dropout_1.forward(Activacion_1.output)

    #Paso los datos de la función de activación a la segunda capa oculta
    Capa_2.forward(dropout_1.output)

    #Paso por la función de activacion softmax
    #Activacion_2.forward(Capa_2.output)

    #print(Activacion_2.output[:5])
    #print(Y[:5])


    #CALCULAR LA PÉRDIDA DE MUESTRA
    #loss_function = Loss_CategoricalCrossEntropy()
    #loss = loss_function.calculate(Activacion_2.output, Y)
    data_loss = loss_activation.forward(Capa_2.output, Y)

    regularization_loss = loss_activation.loss.regularization_loss(Capa_1) + loss_activation.loss.regularization_loss(Capa_2)
    
    loss = data_loss + regularization_loss


    #PRINT LOSS VALUE
    #print('Loss:', loss)

    predicciones = np.argmax(loss_activation.output, axis = 1)

    if len(Y.shape) == 2:
        Y = np.argmax(Y, axis = 1)

    precision = np.mean(predicciones == Y)
    #print('Precisión: ', precision)

    #Print la precisión
    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
              f'acc: {precision:.3f}, ' +
              f'loss: {loss:.3f}, ' + 
              f'data_loss: {data_loss:.3f}, ' +
              f'regularization_loss: {regularization_loss:.3f}, ' +
              f'lr: {optimizer.current_learning_rate:.5f}')

    #Backwards para calcular los gradientes de toda la red
    loss_activation.backwards(loss_activation.output, Y)
    Capa_2.backwards(loss_activation.dinputs)
    
    dropout_1.backwards(Capa_2.dinputs)
    
    Activacion_1.backwards(dropout_1.dinputs)
    Capa_1.backwards(Activacion_1.dinputs)

    #Actualizamos los parámetros de la red con el optimizador
    optimizer.pre_update_params()
    optimizer.update_params(Capa_1)
    optimizer.update_params(Capa_2)
    optimizer.post_update_params()