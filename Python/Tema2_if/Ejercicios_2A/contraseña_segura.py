contraseña = input("Introduce una contraseña: ")
upper_validator = False
lower_validator = False
digit_validator = False
asterisco_validator = False
almohadilla_validator = False

for c in contraseña:
    if c.isupper():
        upper_validator = True

for c in contraseña:    
    if c.islower():
        lower_validator = True

for c in contraseña:
    if c.isdigit():
        digit_validator = True

for c in contraseña:
    if c == '*':
        asterisco_validator = True

for c in contraseña:
    if c == '#':
        almohadilla_validator = True

if upper_validator & lower_validator & digit_validator & asterisco_validator & almohadilla_validator:
    print("La contraseña es segura!")
else:
    print("La contraseña no es segura!")