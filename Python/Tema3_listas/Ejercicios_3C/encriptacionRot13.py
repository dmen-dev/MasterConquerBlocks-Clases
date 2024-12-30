abecedario_latino = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

abecedario_rot13 = ['n','o','p','q','r','s','t','u','v','w','x','y','z','a','b','c','d','e','f','g','h','i','j','k','l','m']

mayus = False

palabra_entrada = input("Introduce una cadena de caracteres: ")

palabra_codif = ''
for letra in palabra_entrada:
    if letra.isupper():
        mayus = True
        #print(mayus)
    for i in range (len(abecedario_latino)):
        if mayus == True:
            if letra.lower() == abecedario_latino[i]:
                palabra_codif += abecedario_rot13[i].capitalize()
        elif letra == abecedario_latino[i]:
            palabra_codif += abecedario_rot13[i]
    mayus = False

print(palabra_codif)


cadena_1 = input("Introduce 1 cadena: ")
cadena_2 = input("Introduce otra cadena: ")

rot_aplicacion = False

for j in range(len(cadena_1)):
    for i in range (len(abecedario_latino)):
        if cadena_1[j] == abecedario_latino[i]:
            if cadena_2[j] == abecedario_rot13[i]:
                rot_aplicacion = True
            else:
                rot_aplicacion = False

if rot_aplicacion == True:
    print("Se aplica ROT13!")
else:
    print("No se aplica ROT13!")
            