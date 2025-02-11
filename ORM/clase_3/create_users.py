from setup import session, User

#1. CREAR UN REGISTRO
#Crear un registro --> CREATE
new_user = User(name = "Dani", age = 28) 
#Agregamos registro
session.add(new_user)
session.commit()

#verificar que el usuario ha sido registrado -->READ
created_user = session.query(User).filter_by(name="Dani").first()

#Agregar multiples usuarios
users_to_create = [
    User(name="Bob",age=30),
    User(name="Maria", age =20),
    User(name="Carlos",age=34),
    User(name="Marcelo",age=56)
]

session.add_all(users_to_create) 
session.commit()

#Verificar los registros creados
all_users = session.query(User).all()
for users in all_users:
    print(f"Usuario: {users.name}, edad = {users.age}")