from setup import Session
from SingleResponsibilityPrinciple import Estudiante, EstudiantesBD
from OpenClosedPrinciple import BecaPorNecesidad, BecaPorRendimiento
from LiskovSubstitutionPrinciple import NotificacionEmail, NotificacionSMS
from InterfaceSegregationPrinciple import ReporteEstudiante, ExportadorReporte
from DependencyInversionPrinciple import SQLAlchemyDB

#Crear la sesion de base de datos
session = Session()
db = SQLAlchemyDB(session)
repositorio = EstudiantesBD(session)

#Agregar estudiantes
estudiante_1 = Estudiante(nombre = "Dani Mendinueta", grado = "A")
estudiante_2 = Estudiante(nombre = "Sergio Machado", grado ="B")
estudiante_3 = Estudiante(nombre = "Ana Martinez", grado = "C")

#db.guardar(estudiante_1)
#db.guardar(estudiante_2)
#db.guardar(estudiante_3)

#Listar estudiantes
estudiantes = repositorio.lista_estudiante()
for estudiante in estudiantes:
    print(f"ID: {estudiante.id} , Nombre: {estudiante.nombre} , Grado: {estudiante.grado}")

#Calcular becas
calculadora_rendimiento = BecaPorRendimiento()
calculadora_necesidad = BecaPorNecesidad()

for estudiante in estudiantes:
    print(f"{estudiante.nombre} - Beca por rendimiento: {calculadora_rendimiento.calcular(estudiante)}")
    print(f"{estudiante.nombre} - Beca por necesidad: {calculadora_necesidad.calcular(estudiante)}")

#Enviar notificaciones
notificacion_email = NotificacionEmail()
notificacion_sms = NotificacionSMS()

for estudiante in estudiantes:
    mensaje = "Felicidades, estas registrado!"
    notificacion_email.enviar(estudiante, mensaje)
    notificacion_sms.enviar(estudiante, mensaje)

#Generar y exportar reporte
generador_reporte = ReporteEstudiante()
exportador_reporte = ExportadorReporte()

for estudiante in estudiantes:
    reporte = generador_reporte.generar_reporte(estudiante)
    exportador_reporte.exportar_pdf(reporte)