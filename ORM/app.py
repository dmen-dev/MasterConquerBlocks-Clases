from database import engine, Base
from models import User
from sqlalchemy.orm import sessionmaker



#Crear las tablas en la base de datos
if __name__ == '__main__':
    print("Insertando usuario en la base de datos...")
    Base.metadata.create_all(bind=engine)
    print("Base de datos lista!")
    try:
    
        print("Insertando datos en la bbdd")

        #Configurando la sessión
        SessionLocal = sessionmaker(autoflush=False, bind = engine)
        db = SessionLocal()

        new_user = User(name = "Dani", age = 28)

        #Agregar usuario
        db.add(new_user)
        db.commit()
        print("Usuario insertado con exito!")
    
    except Exception as e:
        db.rollback()
        print("El usuario no pudo ser insertado")
