import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Capa_densa:
    #INICIACIÓN DE LA CAPA
    def __init__(self, n_inputs, n_neuronas):
        #INICIO DE LOS PESOS ALEATORIOS DE LA CAPA
        self.weights = 0.01 * np.random.rand(n_inputs, n_neuronas)
        #INICIO DE LOS SESGOS DE LA CAPA
        self.biases = np.zeros((1, n_neuronas))
        pass

    def forward(self, inputs):
        #CALCULA LOS OUTPUT DE LA CAPA A TRAVÉS DEL PRODUCTO ESCALAR
        self.output = np.dot(inputs, self.weights) + self.biases
        pass

class Activation_ReLu:

    def forward(self, inputs):

        #CALCULAMOS EL OUTPUT

        self.output = np.maximum(0, inputs)

class Activation_softmax:

    def forward(self, inputs):
        #CALCULO LOS VALORES EXPONENCIALES RESTÁNDOLES EL VALOR MÁXIMO
        exp_values = np.exp(inputs-np.max(inputs, axis = 1, keepdims = True))

        #NORMALIZO LAS PROBABILIDADES
        probabilidades = exp_values / np.sum(exp_values, axis = 1, keepdims = True)

        self.output = probabilidades


def Column_selection(df1):

    #df_limpio = df1[['Tamano','Peso','Dulzor','Textura', 'Jugosidad', 'Madurez', 'Acidez','Calidad']]
    df_limpio = df1.drop('A_id', axis = 1)
    return df_limpio

def Analisis_datos(df2):
    C1 = df2['Tamano']
    
    #print(C1)

    if C1.isna().any():
        #print('Hay nulos')
        pass

##    for dato in df2.iterrows():

##        if pd.isna(dato[1]['Tamano']):
##            #print('True NaN')
##            pass

    #COMPRUEBO LOS VALORES NULOS
    nulos = {}

    for columna in df2.columns:
        nulos[columna] = 0

        for datos in df2.iterrows():
            
            if pd.isna(datos[1][columna]):
                nulos[columna] += 1
    
    #PRINT NULOS POR COLUMNAS

    for columna in df2.columns:

        print('Columna', columna, 
              round(
                  1-nulos[columna]/len(df2[columna]), 5
              )*100, '%')
        
    #PARA COMPROBAR LOS OUTLIERS LOS MÁS FÁCIL ES DIBUJARLO

    for col in df2.columns:
        
        if col!='Calidad':

            plt.figure(figsize=(10,6))
            df2[col].hist(bins=10)
            plt.title("Histograma de "+col)
            plt.xlabel('Valor de intervalo')
            plt.ylabel('Frecuencia')
            plt.show()

#ELIMINAR LOS DATOS NULOS O NaN (Not a Number)
#ELIMINAR DEL MODELO LOS OUTLIERS ---> 10
def Clean_data(df3):

    #print(len(df3))#filas antes
    #df3 = df3.dropna()
    df3.dropna(inplace=True)
    #print(len(df3))#filas después 

    #ELIMINO LOS OUTLIERS
    for col in df3.columns:

        if col!='Calidad':

            df3 = df3[df3[col] < 10] 

    #print(df3)

    mapeo = {'bad':0, 'good':1}
    df3['Calidad'] = df3['Calidad'].map(mapeo)
    #print(df3)  
          

df = pd.read_csv("apple_quality_sp.csv")

#SELECCIONAMOS LAS COLUMNAS QUE NOS INTERESAN DEL MODELO
df_limpio = Column_selection(df)
#print(df_limpio.head())

#REALIZAMOS UN ANÁLISIS DE LOS DATOS
#Analisis_datos(df_limpio)

Clean_data(df_limpio)

#AHORA QUE ESTÁN LOS DATOS LIMPIOS PODEMOS TRABAJAR CON MODELO
Y = df_limpio['Calidad']
df_limpio.drop('Calidad', axis = 1, inplace = True)
X = df_limpio

#CONVERTIMOS NUESTRO DATAFRAME EN UN ARRAY DE NUMPY
Y = np.array(Y)
X = np.array(X)

#DEFINO NUESTRA RED NEURONAL DE 3 CAPAS OCULTA DE 12, 4 Y 6 NEURONAS

#CAPA DE ENTRADA 7 NEURONAS Y CAPA DE SALIDA 2
#DEFINIMOS LA PRIMERA CAPA
Capa1 = Capa_densa(7,12)

#DEFINIMOS LA FUNCIÓN DE ACTIVACIÓN PARA LA CAPA 1
Activation1 = Activation_ReLu()

#DEFINIMOS LA SEGUNDA CAPA Y SU ACTIVACIÓN
Capa2 = Capa_densa(12,4)
Activacion2 = Activation_ReLu()

#DEFINIMOS LA TERCERA CAPA Y SU ACTIVACIÓN
Capa3 = Capa_densa(4,6)
Activacion3 = Activation_softmax()

#DEFINO OPERATIVA DE LA CAPA
Capa1.forward(X)
Activation1.forward(Capa1.output)

Capa2.forward(Activation1.output)
Activacion2.forward(Capa2.output)

Capa3.forward(Activacion2.output)
Activacion3.forward(Capa3.output)   

print(Activacion3.output[:5])