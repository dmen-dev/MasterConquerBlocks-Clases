num_entero = int(input("Introduce un número entero: "))

for i in range (1,num_entero+1):
    for j in range (i):
        print('*', end="")
    print("")
for i in range (num_entero-1, 0, -1):
    for j in range (i):
        print('*', end = "")
    print("")