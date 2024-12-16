genero = input("El alumno es chica o chico: ")
nombre_alumno = input("introduce el nombre del alumno: ")
nombre_alumno = nombre_alumno.capitalize()

letrasChicoA ="ABCDEFGHRSTUVWXYZ"
letrasChicaA ="EFGHIJKLM"

grupoA = False

if genero == "chico":
    for letra in letrasChicoA:
        if nombre_alumno[0] == letra:
            grupoA = True

if genero == "chica":
    for letra in letrasChicaA:
        if nombre_alumno[0] == letra:
            grupoA = True
    
if grupoA:
    print("El alumno corresponde al grupo A")
else:
    print("El alumno corresponde al grupo B")