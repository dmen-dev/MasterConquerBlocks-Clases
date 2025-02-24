import numpy as np

# neurona = w*input + w*input + w*input + b

inputs = [1.5, 3, 2, 1.6]
weights = [0.5, -0.3, 0.5, 1]
bias = 1.05

def neurona():
    output = (
        inputs[0] * weights [0] +
        inputs[1] * weights [1] +
        inputs[2] * weights [2] +
        inputs[3] * weights [3] +
        bias
    )

    print(f"Neurona: {output}")

######################

def neurona_numpy():
    output = np.dot(weights, inputs) + bias
    print(f"Neurona numpy: {output}")

#para vectores
a = np.array([1,2,3])
b = np.array([4,5,6])

producto = np.dot(a,b)
print(f"Producto: {producto}")

A = np.array([[5,6],[7,8],[4,3]])
B = np.array([[1,2],[3,4]])

producto_matrix = np.dot(A,B)
print(f"Producto matrix: {producto_matrix}")

neurona()
neurona_numpy()

