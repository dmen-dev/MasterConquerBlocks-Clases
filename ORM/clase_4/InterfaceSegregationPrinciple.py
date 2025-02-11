#PRINCIPIO DE SEGREGACIÓN DE INTERFACES
#Las clases de orden superior no dependen de métodos que no usan

#Interfaces específicas

class GeneradorReportes:
    def generar_reporte(self, estudiante):
        raise NotImplementedError
    
class ExportadorPDF:
    def exportar_pdf(self, reporte):
        raise NotImplementedError

#Clases que cumplen ISP
class ReporteEstudiante(GeneradorReportes):
    def generar_reporte(self, estudiante):
        return f"Reporte para : {estudiante.nombre} en grado {estudiante.grado}"    
    
class ExportadorReporte(ExportadorPDF):
    def exportar_pdf(self, reporte):
        print(f"Exportando PDF de reporte para: {reporte}")
