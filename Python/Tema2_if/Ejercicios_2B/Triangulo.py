print("Introduce 3 longitudes")
longitud_1 = int(input("Introduce la longitud de la primera pieza: "))
longitud_2 = int(input("Introduce la longitud de la segunda pieza: "))
longitud_3 = int(input("Introduce la longitud de la tercera pieza: "))

Cons_posible = False

if (longitud_1 < longitud_2 + longitud_3) and (longitud_2 < longitud_1 + longitud_3) and (longitud_3 < longitud_1 + longitud_2):
    print("Es posible construir la estructura!")
else:
    print("No es posible construir la estructura!")