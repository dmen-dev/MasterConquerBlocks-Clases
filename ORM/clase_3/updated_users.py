from setup import session, User

#4. ACTUALIZAR REGISTROS
user_to_update = session.query(User).filter_by(name="Dani").first()
if user_to_update:
    user_to_update.age = 29
    session.commit()
    print(f"Usuario actualizado: {user_to_update}, nueva edad: {user_to_update.age}")

#Actualización de multiples registros -> incrementar la edad para todos los usuarios en 1
users_to_update = session.query(User).all()
for user in users_to_update:
    user.age +=1
session.commit()

#Verificar los cambios
updated_users = session.query(User).all()
print("Usuarios actualizados:")
for user in updated_users:
    print(f"{user.name}, {user.age}")

#intentar actualizar usuario inexistente
user_to_update = session.query(User).filter_by(name="NombreNoExiste").first()
if not user_to_update:
    print("El usuario que intentas actualizar no existe")