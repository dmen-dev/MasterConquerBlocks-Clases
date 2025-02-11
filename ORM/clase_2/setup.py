from sqlalchemy import create_engine, Column, Integer, String
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv


load_dotenv()


#URL de BBDD
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"mysql+pymysql://root:{DATABASE_PASSWORD}@localhost:3306/CONQUERBLOCKS"

#crear el motor de la bbdd
engine = create_engine(DATABASE_URL, echo=True)

#clase para definir los modelos
Base = declarative_base()

#configuración de la sesión
#SessionLocal = sessionmaker(autoflush=False, bind = engine)
Session = sessionmaker(bind = engine)
#abrir sesion
session = Session()

#Definición de un modelo
class User(Base):
    __tablename__ = "Users"
    id = Column(Integer, primary_key = True)
    name = Column(String)
    age = Column(Integer)

#Crear tabla
try:
    Base.metadata.create_all(engine)
    print("Base de datos y tabla creadas exitosamente")
except:
    print("No se pudo crear la base de datos")

#Crear un registro --> CREATE
new_user = User(name = "Dani", age = 28) 
#Agregamos registro
session.add(new_user)
session.commit()

#verificar que el usuario ha sido registrado -->READ
created_user = session.query(User).filter_by(name="Dani").first()
print(f"Usuario creado: {created_user.name}, Edad: {created_user.age}")

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

#2. MANEJO EXCEPCIONES
try:
    new_user = User(name="Samanta", age = 20)
    session.add(new_user)
    session.commit()
except Exception as e:
    print(f"Error al crear el usuario {e}")

#3. LEER REGISTROS
#Operacion filtrado simple
users = session.query(User).filter(User.age > 25).all()
print("Usuarios mayores de 25:")
for user in users:
    print(f"{user.name}, {user.age}")

#Operacion filtrado multiple
filtered_users_2 = session.query(User).filter(User.age>25, User.age<30).all()
print("Usuarios entre 20 y 30")
for users in filtered_users_2:
    print(f"Usuario: {users.name}, edad = {users.age}")

#Contar registros
total_filtered_users = (session.query(User).filter(User.age>25,User.age<30).count())
print(f"Total de usuarios en la base de datos: {total_filtered_users}")

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

#5. ELIMINAR UN REGISTRO
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