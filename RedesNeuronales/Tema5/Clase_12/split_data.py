import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv('Titanic-Dataset.csv')

n_splits = 5

kf = KFold(n_splits, shuffle = True, random_state = 42)

for train_index, test_index in kf.split(df):

    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    #print de las divisiones y tamaños

    print('Tamaño del conjunto de entrenamiento: ', train_df.shape)
    print('Tamaño del conjunto de test: ', test_df.shape)