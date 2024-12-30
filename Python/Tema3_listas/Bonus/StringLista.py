string1 = "David Fernandez 12311267A 43527 2 9.1 7.6 2.4\nMaria Garcia 12316487A 43527 2 7.1 8.6 5.4\nJuan Perez 6477829236A 43527 2 8.1 8.5 8.4\n"

listaAlumnos = []

separarDistAlumnos = string1.split("\n")
#print(separarDistAlumnos)
for lista in separarDistAlumnos:
    listaAlumnos.append(lista.split(" "))


for alumno in listaAlumnos:
    if(len(alumno) == 8):
        notaMedia = (float(alumno[5]) + float(alumno[6]) + float(alumno[7]))/3
        print(f"Alumno con DNI: {alumno[2]} tiene una nota media: {notaMedia}")
