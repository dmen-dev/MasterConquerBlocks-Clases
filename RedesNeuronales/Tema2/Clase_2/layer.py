import numpy as np

#input de la capa anterior viene de 4 neuronas

inputs = [1.5, 3, 2, 1.6]

#nuestra red tiene 3 neuronas

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

#Como son 3 neuronas los sesgos también tienen que ser de dimensión 3
biases = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + biases
print(f"Layer output: {layer_outputs}")