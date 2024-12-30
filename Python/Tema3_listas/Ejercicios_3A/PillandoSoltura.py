#1
lista = [1,2,3,4,5,6,4,3,1]
elementos_duplicados = []
elementos_unicos = []

for i in lista:
    if(lista.count(i)) > 1:
        elementos_duplicados.append(i)
#        for j in range (lista.count(i)):    
#            lista.remove(i)
    else:
        elementos_unicos.append(i)

print(elementos_duplicados)
#print(lista)
print(elementos_unicos)

#2
lista_sum = []
lista_1 = [1,2,3,4,5]
lista_2 = [6,7,8,9]

lista_sum = lista_2 + lista_1
lista_sum.sort()
print(lista_sum)

#3
lista.sort()
print(lista_sum[-2])

#4
