#nombres_usuario = ["juan123", "ana456", "pedro789"]
#contraseñas = ["clave123", "clave456", "clave789"]

usuarioYpassword = [["juan123", "clave123"], ["ana456", "clave456"], ["pedro789", "clave789"]]

nombre_usuario = input("introduce nombre usuario: ")
contraseña = input("introduce contraseña: ")

usuarioOK = False
passOK = False

for usuarioPass in usuarioYpassword:
    if usuarioPass[0] == nombre_usuario and usuarioPass[1] == contraseña:
        usuarioOK = True
        passOK = True
        
        
if usuarioOK == True and passOK == True:
    print("Acceso permitido")
else:
    print("Acceso no permitido")
