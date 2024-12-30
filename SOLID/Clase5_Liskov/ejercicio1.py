class MetodoPagoBase:
    def procesar_pago():
        pass

class MetodoPagoAutomatico(MetodoPagoBase):
    def procesar_pago(self, cantidad):
        pass

class MetodoPagoManual(MetodoPagoBase):
    def procesar_pago(self, cantidad):
        pass

#Metodos de pago automáticos
class PagoTarjeta(MetodoPagoAutomatico):
    def __init__(self, numero_tarjeta):
        self.numero_tarjeta = numero_tarjeta

    def procesar_pago(self, cantidad):
        print(f"Procesando pago automático de {cantidad} usando tarjeta {self.numero_tarjeta}")

class PagoPayPal(MetodoPagoAutomatico):
    def __init__(self, numero_paypal):
        self.numero_paypal = numero_paypal

    def procesar_pago(self, cantidad):
        print(f"Procesando pago automático de {cantidad} usando PayPal cuenta {self.numero_paypal}")

class PagoBitcoin(MetodoPagoAutomatico):
    def __init__(self, direccion_bitcoin):
        self.direccion_bitcoin = direccion_bitcoin

    def procesar_pago(self, cantidad):
        print(f"Procesando pago automático de {cantidad} usando Bitcoin {self.direccion_bitcoin}")

#Métodos de pago manuales        
class PagoCheque(MetodoPagoManual):
    def __init__(self, numero_cheque):
        self.numero_cheque = numero_cheque

    def procesar_pago(self, cantidad):
        print(f"Procesando pago manual de {cantidad} usando cheque {self.numero_cheque}")

def realizar_pago_automatico(metodo_pago: MetodoPagoAutomatico, cantidad):
    metodo_pago.procesar_pago(cantidad)

def realizar_pago_manual(metodo_pago: MetodoPagoManual, cantidad):
    metodo_pago.procesar_pago(cantidad)

#Instanciar las clases
pago_tarjeta = PagoTarjeta("123 456 789 123")
pago_paypal = PagoPayPal ("mi_cuenta@pago.com")
pago_bitcoin = PagoBitcoin("jiofehwarnwek")
pago_cheque = PagoCheque("123456789")

realizar_pago_automatico(pago_tarjeta, 4000)
realizar_pago_automatico(pago_paypal, 4000)
realizar_pago_automatico(pago_bitcoin, 4000)
realizar_pago_manual(pago_cheque, 4000)
