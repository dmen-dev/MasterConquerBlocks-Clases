import numpy as np

#1
array_1 = np.zeros(8)

#2
array_1[:] = 2

#3
array_2 = np.arange(2,11,2)

#4
print(sum(array_2))

#5
array_2_reverted = array_2[::-1]
print(array_2_reverted)

#6
interseccion_1 = np.intersect1d(array_1, array_2)
print("Elementos comunes entre array_1 y array_2: ", interseccion_1)

#7
longArray = int(input("Introduce una longitud para el array: "))
array_usuario = np.zeros(longArray)
array_usuario[:] = 1
print(array_usuario)        
