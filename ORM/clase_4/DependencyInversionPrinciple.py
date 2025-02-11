#PRINCIPIO DE INVERSIÓN DE DEPENDENCIA
#Las clases deben depender de abstracciones y no de implementaciones concretas

#ABSTRACCIÓN DE BASE DE DATOS PARA UNA IMPLEMENTACIÓN INDISTINTA

#Abstraccion para guardar datos
class BaseDeDatos:
    def guardar(self, data):
        raise NotImplementedError
    
#Implementación concreta de BD con SQLALCHEMY
class SQLAlchemyDB(BaseDeDatos):
    def __init__(self, session):
        self.session = session

    def guardar(self, data):
        self.session.add(data)
        self.session.commit()