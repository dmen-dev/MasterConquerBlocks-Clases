precio_usuario = float (input("Introduce el precio: "))

if precio_usuario < 100:
    print("Orden de compra!")
elif precio_usuario <= 150:
    print("Orden de hold!")
elif precio_usuario > 150:
    print("Orden de vender!")