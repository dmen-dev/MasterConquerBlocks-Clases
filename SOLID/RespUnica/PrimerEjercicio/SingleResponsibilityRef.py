class Engine:
    def getRPM(self):
        return 3000 #valor por defecto motor
    
class Vehicle:
    def __init__(self,name,speed,engine):
        self._name = name
        self._speed = speed
        self._engine = engine

    def getName(self):
        return self._name
    
    def getEngineRPM(self):
        return self._engine.getRPM()
    
    def getMaxSpeed(self):
        return self._speed
    
class VehiclePrinter:
    def __init__(self, vehicle):
        self._vehicle = vehicle

    def printInfo(self):
        print("Vehicle: {}, MaxSpeed: {}, RPM: {}".format(
            self._vehicle.getName(),
            self._vehicle.getMaxSpeed(),
            self._vehicle.getEngineRPM()
        ))

class VehiclePersistence:
    def __init__(self,vehicle,db):
        self._vehicle = vehicle
        self._persistence = db
        print("Hey, storing data in ", self._persistence)

if __name__ == "__main__":
    engine = Engine()
    vehicle = Vehicle(name = "car", speed = 200, engine = engine)
    persistence = VehiclePersistence(vehicle = vehicle, db = "SQL")
    printer = VehiclePrinter(vehicle = vehicle)
    printer.printInfo()