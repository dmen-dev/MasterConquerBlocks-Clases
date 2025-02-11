from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

#setup
load_dotenv()
Base = declarative_base()

#Configuracioón de la base de datos
DATABASE_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"mysql+pymysql://root:{DATABASE_PASSWORD}@localhost:3306/estudiantesdb"
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind = engine)

