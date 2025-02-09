from sqlalchemy import create_engine
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
