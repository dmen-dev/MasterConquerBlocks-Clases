'''
Análisis de precios de productos:

Escribir un programa en Python que analice una lista
de precios de productos y determine el precio más alto,
el precio más bajo y el precio promedio de todos los 
productos. Soluciona el ejercicio sin usar las funciones 
max() o min().
'''

# crear la lista de precios
lista_precios = [3.4, 75.3, 23, 52]

#bucle para recorrer la lista
precio_alto = lista_precios[0]
precio_bajo = lista_precios[0]
precio_total = 0

'''
for i in range (len(lista_precios)):
    if lista_precios[i] >= precio_alto:
        precio_alto = lista_precios[i]
    if lista_precios[i] <= precio_bajo:
        precio_bajo = lista_precios[i] 
    precio_promedio = precio_promedio + lista_precios[i]
'''

for precio in lista_precios:
    if precio >= precio_alto:
        precio_alto = precio
    if precio <= precio_bajo:
        precio_bajo = precio
    precio_total += precio


print("Precio más alto es: ", precio_alto)
print("Precio más bajo es: ", precio_bajo)
print("Precio promedio es: ", precio_total/(len(lista_precios)+1))