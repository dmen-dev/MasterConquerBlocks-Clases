numeroPrimo = True
for i in range(1,100):
    if i == 1:
        print(i)
    else:
        for j in range (2,i):
            if i%j == 0:
                numeroPrimo = False
        if numeroPrimo == True:
            print(i)
        else:
            numeroPrimo = True
