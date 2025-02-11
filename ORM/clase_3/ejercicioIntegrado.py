from sqlalchemy import create_engine, Column, Integer, String
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

#configuracion de la base de datos
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
Base.metadata.create_all(engine)

#FUNCIONES CRUD
def create_users(users):
    try:
        session.add_all(users)
        session.commit()
        print("Usuarios creados existosamente")
    except Exception as e:
        session.rollback()
        print(f"Error al crear usuarios: {e}")

def read_users(age_filter = None):
    if age_filter:
        users = session.query(User).filter(User.age>age_filter).all()
        print(f"Usuarios con edad mayor a {age_filter}")
    else:
        users = session.query(User).all()
    
    for user in users:
        print(f"ID: {user.id}, Nombre: {user.name}, Edad: {user.age}")

def update_user(user_id, new_name = None, new_age = None):
    user = session.query(User).filter_by(id=user_id).first()
    if not user:
        print(f"Usuario con ID: {user_id} no encontrado")
        return
    if new_name:
        user.name = new_name
    if new_age:
        user.age = new_age
    session.commit()
    print(f"Usuario con ID: {user_id} actualizado")
    

def delete_users_by_id(user_id):
    user = session.query(User).filter_by(id = user_id).first()

    if not user:
        print(f"Usuario con ID: {user_id} no encontrado")
        return
    session.delete(user)
    session.commit()
    print(f"Usuarion con ID: {user_id} ha sido eliminado")

def report_users():
    total_users = session.query(User).count()
    if total_users == 0:
        print("No hay usuarios en la base de datos")
        return
    avg_age = session.query(User.age).all()
    avg_age = sum(age[0] for age in avg_age) / total_users
    print(f"Total de usuarios: {total_users}")
    print(f"Edad promedio: {avg_age:.2f}")

#Simulación de registro de usuarios
def main():
    #Crear usuarios
    print("Creado usuarios...")
    create_users(
        [
            User(name="Bob",age=30),
            User(name="Maria", age =20),
            User(name="Carlos",age=34),
            User(name="Marcelo",age=56)
        ]
    )

    #Lista de usuarios
    print("\n Listando usuarios...:")
    read_users()

    #Filtrar usuarios mayores a 25
    print("\n Listando usuarios mayores de 25...")
    read_users(age_filter = 25)

    #Actualizar usuario
    print("\n Actualizado usuario con id 65...")
    update_user(user_id = 65 , new_name = "Benito", new_age = 16)

    #Listar usuarios nuevamente
    print("\n Listando usuarios después de la actualización...")
    read_users()

    #Eliminar un usuario
    print("\nEliminando usuario con id 74...")
    delete_users_by_id(user_id = 74)

    #Generamos un reporte
    print("\nGenerando reporte...")
    report_users()

if __name__ == "__main__":
    main()

