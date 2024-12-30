clave = "passw0rd"

clave_introd_1 = input("Introduce la contraseña: ")

if clave_introd_1 == clave:
    print("Bienvenid@")
else:
    clave_introd_2 = input("Vuelve a introducir la contraseña: ")
    if clave_introd_2 != clave: 
        print("ERROR! Ha realizado 2 fallos al introducir la contraseña!")
    else:
        print("Bienvenid@!")
