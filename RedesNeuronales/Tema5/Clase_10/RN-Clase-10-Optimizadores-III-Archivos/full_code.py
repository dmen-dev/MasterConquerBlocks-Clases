import pandas as pd
import numpy as np

# Capa densamente conectada

class Capa_densa:
    # Iniciación de la capa
    def __init__(self, n_inputs, n_neuronas):

        # Iniciamos con unos valores los pesos y los sesgos de la capa

        self.weights = 0.01 * np.random.randn(n_inputs, n_neuronas)

        self.biases = np.zeros((1, n_neuronas)) #iniciamos con 0, se inicia con 0 para que en determinados escenarios las neuronas se activen 
                                                #sin tener el cuenta el sesgo, evitando neuronas muerta
        pass 
    # Pasar valores a la siguiente capa
    def forward(self, inputs):

        self.inputs = inputs #memorizo los valores de los inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        pass 

    def backward(self, dvalues):

        # Gradiente de los parametros
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient de los valores
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activacion_ReLu:

    # Calcular valores
    def forward(self, inputs):
    # Memorizo los valores de los inputs
        self.inputs = inputs
    # Calculo el output desde el input
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):

        #Como tenemos que modificar los valores hacemos uns copia de los mismos
        self.dinputs = dvalues.copy()
        # el gradiente es 0 cuando los valores son negativos porque la neuora esta desactivada
        self.dinputs[self.inputs <= 0] = 0

# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        #memorizo los valores de los inputs
        self.inputs = inputs
        # Obtengo las probabilidades no normalizadas
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #restamos el maximo de los inputs, para evitar que salgan numeros muy grandes Overflow
        # Normalizo las probabilidades en una distribución 0,1
        probabilidades = exp_values / np.sum(exp_values, axis=1,keepdims=True)

        self.output = probabilidades

    # Método backward: Implementa el paso hacia atrás para la función softmax, 
    # calculando el gradiente con respecto a la entrada usando la matriz jacobiana.
        
    def backward(self, dvalues):

        # Creo un array no iniciado
        self.dinputs = np.empty_like(dvalues)

        #En funcion de los inputs y outputs genero los gradientes con la forma adecuada


        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            # El resultado de esta operación, realizada sobre un lote de muestras, 
            # es una lista de las matrices jacobianas, que forman efectivamente una matriz 3D

            #Para poder operar con ella debemos de "Aplanarla"
            single_output = single_output.reshape(-1, 1)

            # Calculo la matriz jacobiana del output
            jacobian_matrix = np.diagflat(single_output) - \
            np.dot(single_output, single_output.T)
            # Calculo el gradiente de la muestra y lo añado a la lista de de gradientes
            self.dinputs[index] = np.dot(jacobian_matrix,single_dvalues)

class Loss:
    # Calcula los datos y la regularizacion de la perdida dada a partir del output de la red y de los valores "verdaderos"

    def calculate(self, output, y):
        # Calculate las perdidas de la muestra
        sample_losses = self.forward(output, y)
        # Calcula la media de la pérdida
        data_loss = np.mean(sample_losses) 
        # Return loss
        return data_loss
    
# Cross-entropy loss -- Especifica para calcular la pérdida cuando es un modelo categórico
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        n_datos = len(y_pred)
        # Utilizamos la funcion Clip data para prevenir la division por 0
        # a función clip de NumPy es utilizada para limitar los valores en un array. 
        # Esta función asegura que los valores del array estén dentro de un intervalo definido por un valor mínimo y máximo
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(n_datos),
            y_true
            ]
        # Mask values - only for one-hot encoded labels --> 
        # one-hot se refiere a que el true value viene en la forma [[0,1],[1,0]] para decir que neurona de salida se debe activar
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped*y_true,
            axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Crea los objetos de funcion de activación y perdida
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    # Aplica la función softmax y luego calcula la pérdida de entropía cruzada.  
    def forward(self, inputs, y_true):

        # Función de activación de la capa de salida
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculamos y devolvemos el valor de la pérdida
        return self.loss.calculate(self.output, y_true)
    
    # Método backward: Realiza el paso hacia atrás para la combinación de softmax y pérdida de entropía cruzada.
    def backward(self, dvalues, y_true):
        # Number of muestras
        samples = len(dvalues)
        # si las etiquetas están en one-hot encoded,
        # las cambiamos a valores discretos
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)      
        # Copiamos los valores para poder modificarlos sin pisar los valores previos
        self.dinputs = dvalues.copy()
        # Calculamos el gradiente
        self.dinputs[range(samples), y_true] -= 1
        # Normalizamos el gradiente
        self.dinputs = self.dinputs / samples     


class Optimizer_SGD: #Con learning Rate decay (decadencia o descenso)
    # Inicio el optimizador - fijo los parametro
    # iniciamos el learning rate en 1.
    def __init__(self, learning_rate=1., decay=0., momentum=0.): 
        #Añado el parametro momento, que representa la fraccion de los gradientes anteriores que utilizamos va entre 0 y 1
        #Cuando fijamos el momento en un valor muy alto, el modelo es como si no aprendiera ya que no sigue realmente el sentido del gradiente
        self.learning_rate = learning_rate #fijamos learning rate
        self.current_learning_rate = learning_rate #Recordamos el learning rate actual
        self.decay = decay #fijamos la tasa de descenso
        self.iterations = 0
        self.momentum=momentum     #NUEVO: fijamos el momento

        # Lo llamamos una vez que hayamos actualizado cualquier parametro
    def pre_update_params(self): #Queda igual
        # Disminuye la tasa de aprendizaje de manera exponencial
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
        
    # Actualizo los parametros
            
    #Aqui se producen los mayores cambios
    def update_params(self, layer):

        #adaptamos la clase para cuando usa o no momento

        if self.momentum: #si usamos el momento

            #si no me hemos iniciado el array momento, lo rellenamos de zeros

            if not hasattr(layer,'weight_momentums'):
                #Creamos el array para los pesos
                layer.weight_momentums=np.zeros_like(layer.weights)
                #creamos el vector para los sesgos
                layer.bias_momentums=np.zeros_like(layer.biases)

            #Actualizo los pesos con el momento
            #Multiplico por el factor de momento y actualizo los gradientes
            weight_updates=self.momentum*layer.weight_momentums - self.current_learning_rate*layer.dweights
            layer.weight_momentums = weight_updates

            #Actualizo los sesgos
            bias_updates=self.momentum*layer.bias_momentums - self.current_learning_rate*layer.dbiases
            layer.bias_momentums = bias_updates

        #si no usamos momento hacemos la misma operacion pero asignada a la variable de actualizacion de pesos y sesgos 
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates= -self.current_learning_rate * layer.dbiases

        #Actualizo los pesos de la capa
            
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Lo llamamos despues de la actualizacion de cualquier parametro 
    def post_update_params(self):
        self.iterations += 1 #Aumento las iteraciones en 1

class Optimizer_Adagrad:

    # iNICIO Los parametros basicos
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon  #Epsilon simplemente lo utilizaremos para evitar dividir por 0

    # Llamar una vez antes de actualizar cualquier parámetro
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *  (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        #Si la capa no tiene iniciado el atributo weight cache lo iniciamos con 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2   #Acumula el cuadrado de los gradientes actuales para los pesos
        layer.bias_cache += layer.dbiases**2 #Acumula el cuadrado de los gradientes actuales para los pesos

        # Actualizamos las tasas de learning rate en base al peso acumulado de los caches
        layer.weights += -self.current_learning_rate *   layer.dweights /  (np.sqrt(layer.weight_cache) + self.epsilon) # le sumo epsilon por si alguno fuera 0
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# RMSprop optimizer
class Optimizer_RMSprop:
    # inicio los parametros del optimizdor
    #Rho es el factor de amortiguacion de la media
    #Como gran diferencia, la tasa de aprendizaje se inicia en valores mucho mas pequeños 0.001 debido a que rho toma mucho momento (inercia) 
    # no hace falta que sea mayor para que siga avanzando

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    # Llamar una vez antes de actualizar cualquier parámetro
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *  (1. / (1. + self.decay * self.iterations))

    # Actualizo parameters
    def update_params(self, layer):

        #Si la capa no tiene iniciado el atributo weight cache lo iniciamos con 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Actualizo los caches con el cuadro de de los gradientes actules (media movil)
        layer.weight_cache = self.rho * layer.weight_cache +  (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2


        #Actualizo los pesos y normalizo con  square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights /  (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate *  layer.dbiases /  (np.sqrt(layer.bias_cache) + self.epsilon)

        # lo llamo despues de actualizar los parametros
    def post_update_params(self):
        self.iterations += 1

# Adam optimizer
class Optimizer_Adam:
    # inicio los parametros del optimizdor
    #Beta 1 controla la tasa de decaimiento del momento de primer orden
    

    # Un valor de B1 cercano a 1 significa que el decaimiento es muy lento y se tiene en cuenta una ventana más grande de gradientes pasados, 
    # lo que proporciona un promedio móvil suavizado. 
    # Un valor más bajo haría que el promedio móvil olvide más rápidamente los gradientes antiguos, dando más peso a los gradientes más recientes. 
    # En la práctica, un valor común para es 0.9.
   
    #Beta 2 controla la tasa de decaimiento del momento de segundo orden
    # un valor de β2  cercano a 1 resulta en un decaimiento más lento, lo que significa que se considera 
    # una ventana más larga de la historia de los gradientes para calcular la variabilidad de estos. 
    # Esto afecta cómo se ajusta adaptativamente la tasa de aprendizaje para cada parámetro basándose en la escala de sus gradientes. 
    
    # Un valor comúnmente utilizado para β2 es 0.999.

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    # Llamar una vez antes de actualizar cualquier parámetro
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *  (1. / (1. + self.decay * self.iterations))
    # Actualizo parameters
    def update_params(self, layer):

        #Si la capa no tiene iniciado el atributo weight cache lo iniciamos con 0s
        if not hasattr(layer, 'weight_cache'):
            #A DIFERENCIA DE LOS METODOS ANTERIORES, TENMOS QUE INICIAR 4 VECTORES 2 DE CACHE Y 2 DE MOMENTOS
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Actualizo los caches con el cuadro de de los gradientes actuales
        layer.weight_momentums = self.beta_1 *  layer.weight_momentums +  (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 *  layer.bias_momentums +   (1 - self.beta_1) * layer.dbiases
        
        #Obtengo el momento corregido
        # self.iteration is 0 at first pass
        #La iteración primera es =0, pero necesitamos que sea 1, sumamos 1 unidad para no dividir por 0

        weight_momentums_corrected = layer.weight_momentums /  (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums /  (1 - self.beta_1 ** (self.iterations + 1))

        # Actualizo el cache con las correcciones
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Obtengo el cache corregido
        weight_cache_corrected = layer.weight_cache /  (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache /  (1 - self.beta_2 ** (self.iterations + 1))

        # Actualizo los pesos y normalizo con  square rooted cache

        layer.weights += -self.current_learning_rate *  weight_momentums_corrected /  (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate *  bias_momentums_corrected /    (np.sqrt(bias_cache_corrected) + self.epsilon)

    # lo llamo despues de actualizar los parametros
    def post_update_params(self):
        self.iterations += 1    

#Leyenda de columnas

#Pclass Ticket class: 1 = 1st, 2 = 2nd, 3 = 3rd
#SibSp No. of siblings / spouses aboard the Titanic
#Parch No. of parents / children aboard the Titanic 
#Passenger fare -- Lo que pago el pasajero


# Cargar el archivo CSV
file_path = 'Titanic-Dataset.csv'
titanic_data = pd.read_csv(file_path)

# Eliminar las columnas 'PassengerId', 'Name'
titanic_data_cleaned = titanic_data.drop(columns=['PassengerId', 'Name','Ticket', 'Cabin', 'Embarked'])


# Transformar los datos de la columna 'Sex' en binario: 0 para 'male' y 1 para 'female'
titanic_data_cleaned['Sex'] = titanic_data_cleaned['Sex'].map({'male': 0, 'female': 1})

# Mostrar las primeras filas del DataFrame resultante
# print(titanic_data_cleaned.head())

#Me doy cuenta que hay algunos datos con nan

titanic_data_cleaned = titanic_data_cleaned.dropna()


#Extraigo los resultados

Resultado=titanic_data_cleaned['Survived']
Y=np.array(titanic_data_cleaned['Survived'])


#Elimino la columna del dataset para

titanic_data_cleaned = titanic_data_cleaned.drop(columns=['Survived'])


#Construyo un array de numpy con los datos limpios

X=np.array(titanic_data_cleaned)


#codigo para generar la red neuronal

#3 En la definición de la capa hemos separado la parte de creación y ejecución.
#Moviendo todas las ejecuciones fwd y back al bucle de entrenamiento


# Creo una capa densamente conectada con 6  entradas (una por cada columna de mi dataset)
# y 10 neuronas
Capa1 = Capa_densa(6, 10)
# Creo la función de activación ReLu para utilizarla en la primera capa oculta
activation1 = Activacion_ReLu()

# Creo una segunda capa oculta con 10 neuronas y salida de 2 parametros (para clasificar si ha sobrevivido o no)
Capa2 = Capa_densa(10, 2)

# # Creamos la función softmax de activación para obtener el output
# activation2 = Activation_Softmax()

# Creamos la función Softmax clasificador convinada con la función de pérdida y de activación
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# Creo el optimizer



# optimizer = Optimizer_Adam(decay=1e-4)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=1e-4)
# optimizer = Optimizer_Adam(learning_rate=0.01, decay=1e-6)


# Entreno en bucle 
for epoch in range(501):

    # Realizamos el forward de los valores de la capa de entrenamiento
    Capa1.forward(X)
    #Realizamos el forward con la funcion de activación (ReLU) que toma los valores de la primera capa densa
    activation1.forward(Capa1.output)

    #Realizamos el forward de la segunda capa, que toma los valores de salida de las funciones de activación de primera capa 
    Capa2.forward(activation1.output)
    # Make a forward pass through activation function
    # it takes the output of second dense layer here
    #Realizamos el forward con la funcion de activación (Softmax) que toma los valores de la segunda capa densa
    # activation2.forward(Capa2.output)

    #Calculo la pérdida utilizando la funcion Activation_Softmax_Loss_CategoricalCrossentropy
    #comparando la salida de la capa 2 con respecto a los valores esperados reales

    loss = loss_activation.forward(Capa2.output, Y)




    # Calculato la precisión utilizando los outputs de la Activación2 (softmax) y de los valores verdaderos
    # calculo los valores a lo largo del primer eje
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(Y.shape) == 2:
        Y = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == Y)
    # Print la precisión

    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f} ' +
        f'lr: {optimizer.current_learning_rate:.5f}')

    # Backward para calcular los gradientes de toda la red
    loss_activation.backward(loss_activation.output, Y)
    Capa2.backward(loss_activation.dinputs)
    activation1.backward(Capa2.dinputs)
    Capa1.backward(activation1.dinputs)

    # Actualizo weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(Capa1)
    optimizer.update_params(Capa2)
    optimizer.post_update_params()

    #Podemos ver que la precision no mejora y que se queda en un mínimo local