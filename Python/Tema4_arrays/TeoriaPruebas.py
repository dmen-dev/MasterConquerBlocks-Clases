import numpy as np
a = np.zeros((3,3), dtype = np.int64)
a[:] = 2
b = np.arange(1,10).reshape((3,3))
print(a)
print(b)
print('-----------')
matrix_multi = np.matmul(a,b)
#matrix_multi = a.dot(b)
#matrix_multi = a @ b
print(matrix_multi)