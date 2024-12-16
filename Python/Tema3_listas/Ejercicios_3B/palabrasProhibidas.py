listaAleatoria = ["define","cinco", "palabras", "prohibidas", "aleatorias"]
listaLetrasProhibidas = ["j", "h", "a"]

listaFiltrada = []

for palabra in listaAleatoria:
    if palabra.count(listaLetrasProhibidas[0]) == 0  and palabra.count(listaLetrasProhibidas[1]) == 0 and palabra.count(listaLetrasProhibidas[2]) == 0:
        listaFiltrada.append(palabra)


print(listaFiltrada)