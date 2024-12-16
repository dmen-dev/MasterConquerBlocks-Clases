password = "contraseña"

correcta = False

while correcta == False:
    pass_usuario = input("Introduce la contraseña: ")
    if pass_usuario == password:
        correcta = True
        print("Ha introducido la contraseña correcta!")
    else:
        print("La contraseña introducida no es correcta")