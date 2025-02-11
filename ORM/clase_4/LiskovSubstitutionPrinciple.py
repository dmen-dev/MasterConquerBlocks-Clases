#SUSTITUCIÓN de LISKOV
#Clases hijas pueden sustituir a las clases padres

class Notificacion:
    def enviar(self, estudiante, mensaje):
        raise NotImplementedError
    
#Subclase que respete LiskovPrinciple
class NotificacionEmail(Notificacion):
    def enviar(self, estudiante, mensaje):
        print(f"Email enviado a {estudiante.nombre}: {mensaje}")

class NotificacionSMS(Notificacion):
    def enviar(self, estudiante, mensaje):
        print(f"SMS enviado a {estudiante.nombre}: {mensaje}")