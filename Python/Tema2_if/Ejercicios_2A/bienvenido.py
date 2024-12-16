nombre = input("Introduce nombre de usuario:")
nombre_sinpunto = nombre.replace (".",'')
nombre_sinalmoadilla = nombre_sinpunto.replace("#",'')
nombre_low = nombre_sinalmoadilla.lower()

"""
if nombre_low == "alejandro":
    print("Bienvenido Alejandro!")
elif nombre_low == "naomi":
    print("Bienvenida Naomi")
elif nombre_low == "sergio":
    print("Bienvenido Sergio")
else:
    print("Bienvenido")
"""

match nombre_low:
    case "alejandro":
        print("Bienvenido Alejandro")
    case "naomi":
        print("Bienvenida Naomi")
    case "sergio":
        print("Bienvenido Sergio")
    case default:
        print("Bienvenido")


