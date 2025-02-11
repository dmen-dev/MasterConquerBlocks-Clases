from setup import session, User

#5. ELIMINAR UN REGISTROA
user_to_delete = session.query(User).filter_by(name="Bob").first()
if user_to_delete:
    session.delete(user_to_delete)
    session.commit()
    print(f"El usuario eliminado: {user_to_delete.name}")
#Eliminar registroS con un criterio
users_to_delete = session.query(User).filter(User.age > 28).all()
for user in users_to_delete:
    session.delete(user)
session.commit()

#Verificar los cambios
not_deteled_users = session.query(User).all()
print("Usuarios restantes:")
for user in not_deteled_users:
    print(f"Users: {user.name}, Edad: {user.age}")