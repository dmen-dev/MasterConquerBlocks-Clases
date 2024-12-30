precios_producto = [30, 9.8, 42.5, 32.6, 71.5, 44, 21.2, 53.2, 25.3, 57.8]
unidades_vendidas = [3, 1, 0, 0, 7, 2, 0, 0, 4, 0]

total_ventas = 0
dinero_producto = []
dinero_total = 0

for i in range (len(precios_producto)):
    total_ventas += unidades_vendidas[i]
    dinero_total += precios_producto[i] * unidades_vendidas[i]
    dinero_producto.append(precios_producto[i]*unidades_vendidas[i])

print("Total ventas: ", total_ventas)
print("Dinero total: ", dinero_total)
print("Dinero por producto: ", dinero_producto)