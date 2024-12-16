numero_1 = int(input ("Inserte el primer numero: "))
numero_2 = int(input ("Inserte el segundo numero: "))
numero_3 = int(input ("Inserte el tercer numero: "))
numero_4 = int(input ("Inserte el cuarto numero: "))

numero_mayor = 0

if numero_1 > numero_mayor:
    numero_mayor = numero_1
if numero_2 > numero_mayor:
    numero_mayor = numero_2
if numero_3 > numero_mayor:
    numero_mayor = numero_3
if numero_4 > numero_mayor:
    numero_mayor = numero_4

print("El mayor número de los cuatro es el: ", numero_mayor)

