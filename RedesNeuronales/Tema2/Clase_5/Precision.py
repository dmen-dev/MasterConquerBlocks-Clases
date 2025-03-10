import numpy as np

#VALORES DE SALIDA
softmax_output = np.array(
    [
        [0.7, 0.3],
        [0.55, 0.45],
        [0.02, 0.98]
    ]
)

#VALORES ESPERADOS
Y_true = np.array([0, 1, 1])

#CALCULA EL MÁXIMO POR CADA UNA DE LAS SALIDAS
#predicciones = np.argmax(softmax_output, axis = 1)

#if len(Y_true.shape) == 1:

#    Y_true = np.argmax(Y_true)
#    print(Y_true)

if len(softmax_output.shape) == 2:

    predicciones = np.argmax(softmax_output, axis = 1)

elif len(softmax_output.shape) == 1:

    predicciones = np.argmax(softmax_output)    

precision = np.mean(predicciones == Y_true)

print('Precision: ', precision)