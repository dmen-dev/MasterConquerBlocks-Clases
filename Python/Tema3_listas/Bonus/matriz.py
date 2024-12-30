#M = [[2,5,3],[6,1,8],[7,5,4]]
M = [[4,2,3],[4,5],[6,8,2]]

numLista = 0
matriz = True
sumaFila = 0
sumaMaxFila = 0
numFila = 0
numMaxFila = 0

#Comprobar si es Matriz
numLista = len(M)
if numLista != 0:
    for lista in M:
        if len(lista) != len(M[0]):
            matriz = False
        
#Ejercicio 1
if matriz == True:
    for lista in M:
        for n in lista:
            sumaFila += n
        if sumaFila > sumaMaxFila:
            sumaMaxFila = sumaFila
            numMaxFila = numFila

        numFila += 1

    print(f"L1 = {M[numMaxFila]}")
else:
    print(f"L1 = []")   

#Ejercicio2
sumaColumna = 0
sumaMaxColumna = 0
maxColumna = 0
columna = []
if matriz == True:
    for j in range (0,len(M[0])):
        for fila in M:
            sumaColumna += fila[j]
        if sumaColumna > sumaMaxColumna:
            sumaMaxColumna = sumaColumna
            maxColumna = j
        sumaColumna = 0
    
    for fila in M:
        columna.append(fila[maxColumna])

    print(f"L2 = {columna}")
else:
    print(f"L2 = []")
