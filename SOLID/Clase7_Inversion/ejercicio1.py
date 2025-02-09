from abc import ABC, abstractmethod

#Abstracción para el servicio de notificación (interface)
class Notificador(ABC):
    @abstractmethod
    def enviar(self, mensaje:str):
        pass

#Implementación del servicio de notificación para correo electrónico
#Clase BajoNivel
class EmailNotificador(Notificador):
    def enviar(self, mensaje:str):
        print(f"Enviando email: {mensaje}")

#Implementación del servición de notificación para SMS
#Clase BajoNivel --> Incluye detalles
class SMSNotificador(Notificador):
    def enviar(self, mensaje:str):
        print(f"Enviando mensaje: {mensaje}")

#Clase o modulo de AltoNivel que maneja la lógica de negocios
class Aplicacion:
    def __init__(self, notificador:Notificador):
        self.notificador = notificador

    def enviar_notificacion(self, mensaje:str):
        self.notificador.enviar(mensaje)
        print(f"Notificación enviada correctamente!")

#MODO DE USO
email_notificador = EmailNotificador()
app_con_email = Aplicacion(email_notificador)
app_con_email.enviar_notificacion("Este es un mensaje de prueba de correo electrónico!")

sms_notificador = SMSNotificador()
app_con_sms = Aplicacion(sms_notificador)
app_con_sms.enviar_notificacion("Este es un mensaje de prueba de SMS!")