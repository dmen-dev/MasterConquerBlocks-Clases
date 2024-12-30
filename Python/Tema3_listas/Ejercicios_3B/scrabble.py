palabras = ["A5", "N4", "I5","O8", "P9", "B2"]

puntuacionMax = 0

for letra in palabras:
    puntuacionMax += int(letra[1])

print(puntuacionMax)