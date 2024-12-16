#1
numeros = [1,2,3,4,5,6,7,8,9,10]
print(numeros)

#2
num_par_inv = []
for num in numeros[::-1]:
    if num%2 == 0:
        num_par_inv.append(num)
print(num_par_inv)

#3
for num in numeros:
    print(num**2)

#4
#Done

#5
print(min(numeros))

#6
print(max(numeros))

#7 bucle
suma = 0
for num in numeros:
    suma = suma + num
print(suma)
#7 sin bucle
print(sum(numeros))

#8
print(numeros.index(8))
print(num_par_inv.index(8))