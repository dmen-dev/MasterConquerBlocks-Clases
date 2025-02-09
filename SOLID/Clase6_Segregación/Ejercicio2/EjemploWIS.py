#using PROTOCOL
from typing import Protocol

class IOperacionFinanciera(Protocol):
    def depositar(self,amount:float)->None:... #3 puntos indican que pase

    def retirar(self, amount:float)->None:...

    def transferir(self, amount:float)->None:...

class CuentaAhorros:
    def depositar(self, amount:float)->None:
        print(f"Depositando {amount} a la cuenta de ahorros!")

    def retirar(self, amount:float)->None:
        print(f"Retirando {amount} de la cuenta de ahorro!")

    def transferir(self, amount:float,a_cuenta:str)->None:
        raise NotImplementedError("La cuenta de ahorros no puede transferir")
        
class CuentaCorriente:
    def depositar(self, amount:float)->None:
        print(f"Depositando {amount} a la cuenta corriente!")

    def retirar(self, amount:float)->None:
        print(f"Retirando {amount} de la cuenta corriente!")

    def transferir(self, amount:float, a_cuenta:str)->None:
        print(f"Transifiriendo {amount} a la cuenta corriente {a_cuenta}")

cuentaAhorros = CuentaAhorros()
cuentaAhorros.depositar(1343.3)
cuentaAhorros.retirar(123.2)

cuentaCorriente = CuentaCorriente()
cuentaCorriente.depositar(12312.2)
cuentaCorriente.retirar(1235)
cuentaCorriente.transferir(1238,"1232183129312")