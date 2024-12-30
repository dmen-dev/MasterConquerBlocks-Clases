import numpy as np

#9. crea un arrays llenos de 1s con una longitud dada por el usuario
longitud = int(input("Introduce longitud del array: "))
array_ones = np.ones(longitud)
print(array_ones)


#10. Cambia la forma del array para que tenga una estructura de tipo (filas, columnas)
filas = int(input("Introduce la cantidad de filas: "))
columnas = int(input("Introduce el número de columnas: "))

if (filas * columnas == longitud):
    nuevo_array = np.reshape(array_ones, (filas, columnas))
    print(nuevo_array)

else:
    print("La cantidad de filas y columnas no es errónea!")


#11. Crea una "matriz identidad" con la misma forma que el array anterior (filas, columnas)
if (filas * columnas == longitud):
    if(filas == columnas):
        array_identidad = np.identity(filas)
        print(array_identidad)
    else:
        print("No es posible tener un array identidad!")

else:
    print("La cantidad de filas y columnas no es errónea!")


#12. Concatena ambas estructuras horizontalmente y verticalmente
#(Pista: Investiga el funcioamiento de concatenate() y de vstack() y hstack() de numpy)
if (filas * columnas == longitud):
    if(filas == columnas):
        concat_horizontal = np.concatenate((nuevo_array, array_identidad), axis = 1)
        #concat_horizontal = np.hstak(nuevo_array, array_identidad)
        print(concat_horizontal)
        concat_vertical = np.concatenate((nuevo_array, array_identidad), axis = 0)
        #concat_vertical = np.vstack(nuevo_array, array_identidad)
        print(concat_vertical)
    else:
        print("No es posible tener un array identidad!")

else:
    print("La cantidad de filas y columnas no es errónea!")