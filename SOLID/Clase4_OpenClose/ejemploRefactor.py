from abc import ABC, abstractmethod

#clase maestra
class ManejadorPedidos(ABC):
    @abstractmethod
    def realizar_pedido(self, detalles):
        pass

class PedidoParaLlevar(ManejadorPedidos):
    def realizar_pedido(self, detalles):
        print(f"Procesando pedido para llevar: {detalles}")
        
class PedidoLocal(ManejadorPedidos):
    def realizar_pedido(self, detalles):
        print(f"Procesando pedido para comer en el local: {detalles}")

class PedidoEntregaADomicilio(ManejadorPedidos):
    def realizar_pedido(self, detalles):
        print(f"Procesando pedido para entrega a domicilio: {detalles}")

class PedidoEspecial(ManejadorPedidos):
    def realizar_pedido(self, detalles):
        print(f"Procesando pedido especial: {detalles}")

class Restaurante:
    def __init__(self, nombre) ->None:
        self.nombre = nombre
        self.manejadores_pedido = []

    def registrar_pedidos(self, tipo_pedido):
        self.manejadores_pedido.append(tipo_pedido)

    def realizar_pedido(self,tipo_pedido, detalles):
        tipo_pedido.realizar_pedido(detalles)

restaurante = Restaurante("Mi restaurante de pastas")
restaurante.registrar_pedidos(PedidoParaLlevar())
restaurante.registrar_pedidos(PedidoEspecial())

restaurante.realizar_pedido(PedidoParaLlevar(), "Plato de pasta grande")
restaurante.realizar_pedido(PedidoEspecial(),"Plato especial de mariscos")