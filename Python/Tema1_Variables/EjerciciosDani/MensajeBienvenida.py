string1 = "estas usando python"

#Mensaje sin modificaciones
print("Introduce el nombre de usuario:")
string2 = str(input())

#Quitar punto si se introduce en medio de nombre
pos = string2.find('.')

if pos >= 0:
    string4 = string2[0:pos] + string2[pos+1: len(string2)]
    #print(string4)
else:
    string4 = string2

string3 = "Hola, " + string4 + ", estas usando python!"
print(string3)

#Mensaje upperCase
print(string3.upper())

#Mensaje lowerCase
print(string3.lower())

#Nombre formato correcto, Fernando (bien), FerNaNdo (mal)
string4 = "Hola, " + string4.title() + ", estas usando python!"
print(string4)

