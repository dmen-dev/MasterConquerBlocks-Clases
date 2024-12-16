edad = int(input("Introduce la edad: "))
salario_mes = float(input("Introduce salario mensual: "))

if edad >= 18 and salario_mes > 1000:
    if salario_mes*12 < 15000:
        print("Tipo impositivo 5%")
    elif salario_mes*12 <= 25000:
        print("Tipo impositivo 15%")
    elif salario_mes*12 <= 35000:
        print("Tipo impositivo 20%")
    elif salario_mes*12 <= 60000:
        print("Tipo impositivo 30%")
    else:
        print("Tipo impositivo 45%")
else:
    print("No es susceptible de realizar la declaracion")