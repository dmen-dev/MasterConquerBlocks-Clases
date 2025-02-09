from typing import Protocol

class IDepositar(Protocol):
    def depositar(self, amount:float)->None:...

class IRetirar(Protocol):
    def retirar(self,amount:float)->None:...

class ITransferir(Protocol):
    def transferir(self, amount:float)->None:...

class CuentaAhorros:
    def depositar(self, amount:float)->None:
        print(f"Depositando {amount} a la cuenta de ahorros!")

    def retirar(self, amount:float)->None:
        print(f"Retirando {amount} de la cuenta de ahorro!")

#    def transferir(self, amount:float,a_cuenta:str)->None:
#       raise NotImplementedError("La cuenta de ahorros no puede transferir")
        
class CuentaCorriente:
    def depositar(self, amount:float)->None:
        print(f"Depositando {amount} a la cuenta corriente!")

    def retirar(self, amount:float)->None:
        print(f"Retirando {amount} de la cuenta corriente!")

    def transferir(self, amount:float, a_cuenta:str)->None:
        print(f"Transifiriendo {amount} a la cuenta corriente {a_cuenta}")

def realizar_pago(cuenta:ITransferir, amount:float)->None:
    cuenta.transferir(amount, "ABKJDASFD")

cuentaAhorros = CuentaAhorros()
#cuentaAhorros.depositar(1343.3)
#cuentaAhorros.retirar(123.2)
#realizar_pago(cuentaAhorros,30)

cuentaCorriente = CuentaCorriente()
#cuentaCorriente.depositar(12312.2)
#cuentaCorriente.retirar(1235)
#cuentaCorriente.transferir(1238,"1232183129312")
realizar_pago(cuentaCorriente,30)