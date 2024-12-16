ventas = [120, 80, 140, 200, 75, 100, 180, 220, 160, 110, 90, 120, 170, 190, 250, 300, 95, 110, 140, 180, 200, 160, 120, 80, 170, 150, 210, 190, 230, 250]

dias_semana = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

venta_dias_semana = [0,0,0,0,0,0,0]

cont_7 = 0

for venta in ventas:
    if cont_7 == 7:
        cont_7 = 0
    venta_dias_semana[cont_7] += venta
    cont_7 +=1

print(venta_dias_semana)