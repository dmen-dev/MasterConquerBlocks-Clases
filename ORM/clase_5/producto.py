from sqlalchemy import Column, Integer, String, Float, Boolean, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

Base = declarative_base()

class Producto(Base):
    __tablename__ = "productos"
    id = Column (Integer, primary_key = True)
    nombre = Column(String(255), nullable = False)
    tipo = Column(String(255), nullable = False)
    precio = Column ( Float, nullable = False)
    disponible = Column (Boolean, default = True)

    def __repr__(self):
        print(
            f"Producto: id={self.id}, nombre={self.nombre}, tipo={self.tipo}, precio={self.precio}, disponible={self.disponible}"
        )

#Configuracion inicial de la base de datos
load_dotenv()
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"mysql+pymysql://root:{DATABASE_PASSWORD}@localhost:3306/orm_clase_5"

engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
