import pandas as pd
import numpy as np

class Capa_densa:
    #incio de la capa
    def __init__(self, n_inputs, n_neuronas):
        #inicio los pesos de la capa con aleatorio
        self.weights = 0.01 * np.random.rand(n_inputs, n_neuronas)
        #inicio de los sesgos de la capa
        self.biases = np.zeros((1, n_neuronas))
        pass

    def forward(self, inputs):
        #calcula los output de la capa a través del producto escalar
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

class Activation_ReLu:

    def forward(self, inputs):

        #calculamos el output

        self.output = np.maximum(0, inputs)

class Activation_softmax:

    def forward(self, inputs):
        #calculo los valores exponenciales restandoles el valor maximo
        exp_values = np.exp(inputs-np.max(inputs, axis = 1, keepdims = True))

        #normalizo las probabilidades
        probabilidades = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

        self.output = probabilidades

class Loss:
    def calculate(self, output, y):
        # CALCULAR LA PÉRDIDA DE MUESTRA
        perdida_muestra = self.forward(output, y)
        # CALCULAR LA PÉRDIDA MEDIA
        perdida_media = np.mean(perdida_muestra)
        return perdida_media

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
Y = np.array(Resultado)


#Elimino la columna del dataset para
titanic_data_cleaned = titanic_data_cleaned.drop(columns = ['Survived'])

#Construyo un array de numpy con los datos limpios
X = np.array(titanic_data_cleaned)
#print(X)

#quitar los valores no completaos, aquellos  como nan


#los inputs son 6 neuronas, porque son 6 las columnas que tenemos en el dataset y 10 neuronas de salida (las de salida nos inventamos)
#Creamos primera capa oculta
Capa_1 = Capa_densa(6, 10)

#Creamos funcion de activación
Activacion_1 = Activation_ReLu()

#Creamos segunda capa oculta
#Se ponen 2 output ya que necesitamos saber si vive o muere
Capa_2 = Capa_densa(10, 2)
Activacion_2 = Activation_softmax()

#Iniciamos la red con valores del dataframe. Como entrada todos los datos del titanic habiendo eliminado la columna de supervivientes
Capa_1.forward(X)

#Funcion de activacion sobre output de capa 1
Activacion_1.forward(Capa_1.output)

#Paso los datos de la función de activación a la segunda capa oculta
Capa_2.forward(Activacion_1.output)

#Paso por la función de activacion softmax
Activacion_2.forward(Capa_2.output)

#print(Activacion_2.output[:5])
#print(Y[:5])


#CALCULAR LA PÉRDIDA DE MUESTRA
loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(Activacion_2.output, Y)

#PRINT LOSS VALUE
print('Loss:', loss)

Perdida_maxima =  999999
Mejor_Capa1_pesos = Capa_1.weights.copy()
Mejor_Capa1_biases = Capa_1.biases.copy()
Mejor_Capa2_pesos = Capa_2.weights.copy()
Mejor_Capa2_biases = Capa_2.biases.copy()

for iteracion in range (100000):

    #ACTUALIZO LOS PESOS Y LOS BIAS DE FORMA ALEATORIA
    Capa_1.weights = 0.05 * np.random.randn(6, 10)
    Capa_1.biases = 0.05 * np.random.randn(1, 10)
    Capa_2.weights = 0.05 * np.random.randn(10, 2)
    Capa_2.biases = 0.05 * np.random.randn(1, 2)

    #HAGO LAS FUNCIONES FORWARD DE LAS CAPAS
    Capa_1.forward(X)
    Activacion_1.forward(Capa_1.output)
    Capa_2.forward(Activacion_1.output)
    Activacion_2.forward(Capa_2.output)

    #CALCULO LA PÉRDIDA DE MUESTRA
    perdida = loss_function.calculate(Activacion_2.output, Y)

    #CALCULO DE LA PRECISION
    predicciones = np.argmax(Activacion_2.output, axis = 1)
    precision = np.mean(predicciones == Y)

    #SI LA PÉRDIDA ES MENOR QUE LA PÉRDIDA MÁXIMA
    if perdida < Perdida_maxima:
        
        print("Se fijan nuevos parámetros en la red, en la iteracción", iteracion, "Pérdida:", perdida, "Precisión:", precision)

        Mejor_Capa1_pesos = Capa_1.weights.copy()
        Mejor_Capa1_biases = Capa_1.biases.copy()
        Mejor_Capa2_pesos = Capa_2.weights.copy()
        Mejor_Capa2_biases = Capa_2.biases.copy()
        Perdida_maxima = perdida


