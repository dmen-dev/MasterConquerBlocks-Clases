numero_1 = int(input("Introduce un numero: "))
numero_divisor = int(input("Introduce el divisor: "))

if numero_divisor == 0:
    print("ERROR: El divisor introducido es erróneo, al ser 0!")
else:
    resultado = int(numero_1 / numero_divisor)
    restante = numero_1 % numero_divisor
    print("Resultado: ", resultado, " y el restante es ", restante)
