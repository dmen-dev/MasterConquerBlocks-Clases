import pandas as pd
import numpy as np

#dataset del titanic
#https://www.kaggle.com/datasets/yasserh/titanic-dataset/data

#Leyenda columnas
#Pclass Ticket class:1 = 1st, 2 = 2nd, 3 = 3rd
#SibSp No. of siblings / spouses aboard the Titanic (si tienes esposa/o)
#Parch No. of parents / children aboard the Titanic (si tienes padre/madre o hijo/a)
#Passenger fare -- Lo que pago el pasajero

df = pd.read_csv("Clase_3/Titanic-Dataset.csv")

#print(df.head())
#print(df.columns)

df_limpio = df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'])

df_limpio ['Sex'] = df_limpio['Sex'].map({'male':0, 'female':1})

#print(df_limpio['Sex']) 

Y = np.array(df_limpio['Survived'])

df_limpio = df_limpio.drop(columns = ['Survived'])
#quitar los valores no completaos, aquellos  como nan
df_limpio = df_limpio.dropna()
X = np.array(df_limpio)

#print(Y)
#print(X)

