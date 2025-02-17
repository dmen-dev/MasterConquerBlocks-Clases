from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from producto import Base, engine

class Pedido(Base):
    __tablename__ = "pedidos"
    id = Column(Integer, primary_key = True )
    cliente = Column(String(255), nullable = False)
    fecha = Column(DateTime, default = datetime.utcnow)
    productos = relationship ("PedidoProducto", back_populates = "pedido")
    total = Column (Float, nullable = False, default = 0.0)

    def calcularTotal(self):
        self.total = sum(pp.producto.precio * pp.cantidad for pp in self.productos)

class PedidoProducto(Base):
    __tablename__ = "pedido_producto"
    id = Column(Integer, primary_key=True)
    pedido_id = Column(Integer, ForeignKey("pedidos.id"))
    producto_id = Column(Integer, ForeignKey("productos.id"))
    cantidad = Column(Integer, nullable = False, default = 1)

    pedido = relationship("Pedido",back_populates = "productos")
    producto = relationship("Producto")

Base.metadata.create_all(engine)