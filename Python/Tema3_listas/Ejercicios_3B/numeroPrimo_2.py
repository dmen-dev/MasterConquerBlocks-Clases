numEnteros = [56,21,5,78,32,54,7,9,53]
numPrimoList = []

numPrimo = True

for num in numEnteros:
    for i in range (2,num):
        if num % i == 0:
            numPrimo = False

    if numPrimo == True:
        numPrimoList.append(num)
    
    numPrimo = True

print(numPrimoList)

print(len(numPrimoList))

print(sum(numPrimoList))