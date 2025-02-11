from setup import session, User

#3. LEER REGISTROS
#Operacion filtrado simple
filtered_users = session.query(User).filter(User.age > 25).all()
print("Usuarios mayores de 25:")
for user in filtered_users:
    print(f"{user.name}, {user.age}")

#Operacion filtrado multiple
filtered_users_2 = session.query(User).filter(User.age>25, User.age<30).all()
print("Usuarios entre 20 y 30")
for users in filtered_users_2:
    print(f"Usuario: {users.name}, edad = {users.age}")

#Contar registros
total_filtered_users = (session.query(User).filter(User.age>25,User.age<30).count())
print(f"Total de usuarios en la base de datos: {total_filtered_users}")