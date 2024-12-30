#1
frutas = ["manzana", "plátano", "cereza", "pera", "higo", "frambuesa", "fresa"]
print(frutas)

#2
print("Longitud de la lista de frutas: ", len(frutas))

#3
print("El objeto número 3 de la lista es: ", frutas[2])

#4
frutas[1] = "mora"
print(frutas)

#5
frutas.append("mango")
print(frutas)

#6
frutas.insert(0,"uva")
print(frutas)

#7
for fruta in frutas:
    print(fruta)

#8
ultima_fruta = frutas.pop(-1)
print(ultima_fruta)

#9
for fruta in frutas:
    print(fruta)

#10
for fruta in frutas:
    print(len(fruta))

#11
for fruta in frutas:
    if len(fruta)>5:
        print(fruta)

#12
frutas.remove("cereza")
print(frutas)

#13
frutas.clear()
print(frutas)