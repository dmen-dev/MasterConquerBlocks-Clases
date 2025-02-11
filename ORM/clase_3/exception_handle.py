from setup import session, User

#2. MANEJO EXCEPCIONES
try:
    new_user = User(name="Samanta", age = 20)
    session.add(new_user)
    session.commit()
except Exception as e:
    print(f"Error al crear el usuario {e}")