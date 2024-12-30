print("Introduce 3 números diferentes")
numero_1 = int(input("Introduce el primer numero: "))
numero_2 = int(input("Introduce el segundo numero: "))
numero_3 = int(input("Introduce el tercer numero: "))

if (numero_1 == (numero_2 + numero_3)):
    print(numero_1,"es la suma de ",numero_2, "y ", numero_3)
if (numero_2 == (numero_1 + numero_3)):
    print(numero_2, "es la suma de ", numero_1,"y ", numero_3)
if (numero_3 == (numero_1 + numero_2)):
    print(numero_3, "es la suma de ", numero_1, "y ", numero_2)