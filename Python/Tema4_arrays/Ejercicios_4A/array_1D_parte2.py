import numpy as np

#1
array_1 = np.random.randint(1, 101, size=(15))
print(array_1)

#2
multi_bucle = np.int64 (1)
for val in array_1:
    multi_bucle = multi_bucle * val

print(multi_bucle)
print(np.prod(array_1))

#3
array_2 = np.random.rand(15)
print(array_2)

#4
suma_arrays = array_1 + array_2
print(suma_arrays)

suma_numpy = np.add(array_1, array_2)
print(suma_numpy)

#5
resta_arrays = array_1 - array_2
print(resta_arrays)

resta_numpy = np.subtract(array_1, array_2)
print(resta_numpy)

#6
mult_arrays = array_1 * array_2
print(mult_arrays)

mult_numpy = np.multiply(array_1, array_2)
print(mult_numpy)

#7
print(np.max(array_1))

#8
print(f"mean: {np.mean(array_1)}, median: {np.median(array_1)}, standard deviation: {np.std(array_1)}")