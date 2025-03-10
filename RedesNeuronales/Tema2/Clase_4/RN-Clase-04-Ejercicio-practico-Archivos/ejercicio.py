import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

        self.output = np.dot(inputs, self.weights) + self.biases
        pass 


class Activacion_ReLu:

    # Calcular valores
    def forward(self, inputs):
    # Calculate output values from input
        self.output = np.maximum(0, inputs)


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Obtengo las probabilidades no normalizadas
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #restamos el maximo de los inputs, para evitar que salgan numeros muy grandes Overflow
        # Normalizo las probabilidades en una distribución 0,1
        probabilidades = exp_values / np.sum(exp_values, axis=1,keepdims=True)

        self.output = probabilidades


def Column_selection(df1):

    # df_limpio=df1[['Tamano','Peso','Dulzor','Textura','Jugosidad','Madurez','Acidez','Calidad']]
    df_limpio=df1.drop('A_id',axis=1)

    # print(df_limpio)

    return df_limpio

def Analisis_datos(df2):

    # C1=df2['Tamano']
    C1=df2.Tamano
    # print(C1)

    if C1.isna().any():
        # print('Hay nulos')
        pass
    
    for dato in df2.iterrows():

        if pd.isna(dato[1]['Tamano']):
            # print('True Nan')
            pass

    #Compruebo los valores nulos

    nulos={}

    for columna in df2.columns:
        nulos[columna]=0

        for datos in df2.iterrows():

            if pd.isna(datos[1][columna]):
                nulos[columna]+=1

    # print(nulos) #escribo cuantos nulos hay

    for columna in df2.columns:

        print('Columna',columna,
            round(
                    1-nulos[columna]/len(df2),5
            )*100
        ,'%')

    #Analisis visual de los datos
    for col in df2.columns:

        if col!='Calidad':

            plt.figure(figsize=(10,6))
            df2[col].hist(bins=10)

            #titulo a los ejes

            plt.title('Histograma de '+col)
            plt.xlabel=('Valor del intervalo')
            plt.ylabel=('frecuencia')

            #Mostramos el gráfico

            plt.show()

            #Eliminar los nulos o NaN (Not a Number)
            #Eliminaremos del modelo los outliers -- >=10


def Clean_data(df3):

    #Elmino nulos y NaN
    # print(len(df3)) #Filas antes
    # df3=df3.dropna()
    df3.dropna(inplace=True)
    # print(len(df3)) #Filas despues

    #Elimino los outliers

    for col in df3.columns:
        if col !='Calidad':

            df3=df3[df3[col]<10]

    # print(df3)

    mapeo={'bad':0,'good':1}
    df3['Calidad']=df3['Calidad'].map(mapeo)
    # print(df3)
    return df3
    

df=pd.read_csv('apple_quality_sp.csv')

# print(df.head())

#Seleccionamos las columnas del modelo
df=Column_selection(df)

#Realiamos el analisis de los datos

# Analisis_datos(df)

#Eliminar los nulos o NaN (Not a Number)
#Eliminaremos del modelo los outliers -- >=10

df=Clean_data(df)

Y=df['Calidad']
df.drop('Calidad',axis=1,inplace=True)
X=df

#Convertimos nuestro dataframe en un array de numpy

Y=np.array(Y)
X=np.array(X)

# print(X)

#Defino nuestra red neuronal de 3 capas oculta de 12,4 y 6 neuronas

#Capa de entrada 7 neuronas y capa de salida 2

#Definimos la primera capa
Capa1=Capa_densa(7,12)

#Definimos la funcion de activacion de la capa 1

Activation1=Activacion_ReLu()

#Definimos la segunda capa

Capa2=Capa_densa(12,4)
Activation2=Activacion_ReLu()

#Capa 3

Capa3=Capa_densa(4,2)
Activation3=Activation_Softmax()

#Defino operativa de la capa

Capa1.forward(X)
Activation1.forward(Capa1.output)

Capa2.forward(Activation1.output)
Activation2.forward(Capa2.output)

Capa3.forward(Activation2.output)
Activation3.forward(Capa3.output)

print(Activation3.output[:5])



