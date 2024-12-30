print("Introduce una cantidad en euros:")
cantidadEuros = float(input())
cantidadDolar = cantidadEuros * 1.2
#print("La cantidad introducida es equivalente a", cantidadDolar,"$")

tasaGestion = 0.1 * cantidadDolar
cantidadDolarRestante = cantidadEuros * 1.2 * 0.9
print("Monto de euros recibido es de ",cantidadEuros,"€")
print("Cambio en dolares equivale a ", cantidadDolar,"$")
print("Tasa de gestión que se queda la casa es de ", tasaGestion,"$")
print("Cantidad final que recibe usuario es de ",cantidadDolarRestante,"$")

numeros = [34, 46, 3, 5,6, 33]
numeros.sort()

numeros2 = numeros[-3::]
print(numeros2) 
