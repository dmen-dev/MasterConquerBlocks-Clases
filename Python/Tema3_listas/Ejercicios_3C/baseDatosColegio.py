nombres_estudiantes = [["Dani",7.0, 8.9, 9.3], ["Ane", 6.8, 9.1, 4.9],["Marta", 7.6, 4.6, 7.5]]

nota_media_est = 0
nota_media_clase = 0

for estudiante in nombres_estudiantes:
    for i in range(len(estudiante)): 
        if i > 0:
            nota_media_est += estudiante[i]
    nota_media_est = nota_media_est / 3
    estudiante.append(nota_media_est)
    nota_media_clase += nota_media_est
    nota_media_est = 0
nota_media_clase = nota_media_clase / len(nombres_estudiantes)

print (nombres_estudiantes)