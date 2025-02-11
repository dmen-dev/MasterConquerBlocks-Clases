#PRINCIPIO ABIERTO Y CERRADO
#Una clase de orden superior debe estar cerrada a modificación pero abierta a extension
#Clase que calcule becas
class CalculadoraBeca:
    def calcular(self, estudiante):
        raise NotImplementedError
    
# 1- Extensión para calculo por rendimiento
class BecaPorRendimiento(CalculadoraBeca):
    def calcular(self, estudiante):
        return "Beca completa" if estudiante.grado == "A" else "Beca no aplicable"   

# 2- Extensión para calculo por necesidad
class BecaPorNecesidad(CalculadoraBeca):
    def calcular(self, estudiante):
        return "50 porciento de beca" if estudiante.grado in ["B", "C"] else "Beca no aplicable"