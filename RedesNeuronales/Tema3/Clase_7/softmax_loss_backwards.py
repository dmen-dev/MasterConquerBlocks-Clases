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
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # SOLO PARA VALORES CATEGORICOS
        if len(y_true.shape) == 1:  # ES IGUAL A UN VECTOR
            correct_confidences = y_pred_clipped[range(n_datos), y_true]
        elif len(y_true.shape) == 2:  # ES IGUAL A UNA MATRIZ
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # CALCULAR LA PÉRDIDA NEGATIVA
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods
    
    def backwards(self, dvalues, y_true):

        muestras = len(dvalues)

        etiquetas = len(dvalues[0])

        if len(y_true.shape) == 1: #si es una columna

            y_true = np.eye(etiquetas)[y_true] #eye sirve para crear una matriz identidad, todo zeros y diagonal unos, pero al poner y_true al final se indica la neurona que se activa

            #ejemplo eye si y = [0,2,1]
            # [[1,0,0],
            # [0,0,1],
            # [0,1,0]]
        
        #calculo el gradiente
        self.dinputs = -y_true / dvalues

        #normalizo el gradiente con el numero de muestras
        self.dinputs = self.dinputs / muestras
        
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


#Testeo de las funciones

softmax_outputs = np.array(
    [[0.6, 0.15, 0.25],
     [0.05, 0.65, 0.2],
     [0.1, 0.85, 0.05]]
)

targets = np.array([0, 1, 1])

#probamos los tiempos
#Combinada
softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
softmax_loss.backwards(softmax_outputs, targets)
dvalues1 = softmax_loss.dinputs #gradiente

#por separado
activacion = Activation_softmax()
activacion.output = softmax_outputs
loss = Loss_CategoricalCrossEntropy()
loss.backwards(softmax_outputs, targets)
activacion.backwards(loss.dinputs)
dvalues2 = activacion.dinputs

#print('Gradientes: combinados activación y la de pérdida')
#print(dvalues1)

#print('Gradientes: por separado activación y la de pérdida')
#print(dvalues2)

from timeit import timeit

def f1(): #combinada
    #Combinada
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.backwards(softmax_outputs, targets)
    dvalues1 = softmax_loss.dinputs #gradiente

def f2(): #por separado
    #por separado
    activacion = Activation_softmax()
    activacion.output = softmax_outputs
    loss = Loss_CategoricalCrossEntropy()
    loss.backwards(softmax_outputs, targets)
    activacion.backwards(loss.dinputs)
    dvalues2 = activacion.dinputs

t1 = timeit(lambda: f1(), number = 50000) #ejecutamos la funcion 10000 veces y medimos su tiempo
t2 = timeit(lambda: f2(), number = 50000) #ejecutamos la funcion 10000 veces y medimos su tiempo

print('Ratio de eficiencia: ', t2/t1) #si el ratio es mayor que 1, es más eficiente la función 1, si es menor que 1, es más eficiente la función 2
