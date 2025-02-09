from abc import ABC, abstractmethod

#Interface
class SmartDevice(ABC):
    @abstractmethod
    def turn_on(self)->None:
        pass

    @abstractmethod
    def turn_off(self)->None:
        pass

    @abstractmethod
    def set_temperature(self)->None:
        pass

#Subclases
class SmartLight(SmartDevice):
    def turn_on(self) -> None:
        print("Turning lights ON!")
    
    def turn_off(self) -> None:
        print("Turning lights OFF!")

#This is not allowing to fulfil interface segregation   
    def set_temperature(self, temperature:int) -> None:
       raise NotImplementedError("SmartLight device cannot set temperature!")
    
class SmartTherm(SmartDevice):
    def turn_on(self) -> None:
        print("Turning temperature ON!")
    
    def turn_off(self) -> None:
        print("Turning temperature OFF!")
    
    def set_temperature(self, temperature:int) -> None:
        print(f"Setting temperature {temperature}")

smartLight = SmartLight()
smartLight.turn_on()
smartLight.turn_off()

smartTherm = SmartTherm()
smartTherm.turn_on()
smartTherm.turn_off()
smartTherm.set_temperature(24)