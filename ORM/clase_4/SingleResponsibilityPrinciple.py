# PRINCIPIO DE RESPONSABILIDAD UNICA
#Una clase solo debe tener una sola responsabilidad
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

#clase para representar entidad de estudiante
class Estudiante(Base):
    __tablename__ = "estudiante"
    id = Column(Integer, primary_key = True)
    nombre = Column(String, nullable = False)
    grado = Column(String, nullable = False)

#clase para manejar operaciones de bbdd del respositorio del estudiante
#Operaciones: agregar estudiantes, listar estudiantes --> CREATE / READ
class EstudiantesBD:
    def __init__(self, session):
        self.session = session
    
    def agregar_estudiante(self, estudiante):        
        try:
            self.session.add(estudiante)
            self.session.commit()
            print("Estudiante {estudiante} agregado correctamente")
        except:
            print("No se pudo agregar al estudiante")

    def lista_estudiante(self):
        return self.session.query(Estudiante).all()
    