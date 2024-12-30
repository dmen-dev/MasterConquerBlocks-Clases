'''
Pide a usuario 4 números y devuelve
los números introducidos en orden ascendente
y el número mayor.
Para ello puedes usar listas y bucles
'''

#inicializar lista de números
lista_numeros = []

#crear un bucle para ir pidiendo números y añade a lista
for i in range (4):
    numero = int(input ("Introducte un número:"))
    lista_numeros.append(numero)

#output: lista con 4 números
print(lista_numeros)

#output: imprime el número más alto
print("El número más alto es: ",max(lista_numeros))

#output: imprime la lista en orden ascendente
print("Los números introducidos en orden ascendente son: ", sorted(lista_numeros))