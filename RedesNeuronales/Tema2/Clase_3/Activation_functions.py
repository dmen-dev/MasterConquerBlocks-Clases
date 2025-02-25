import numpy as np

def step_fun(inputs):

    output = []

    for i in inputs:
        if i>0:
            output.append(1)
        else:
            output.append(0)

    return output

A = [-1, 0, 0.1, 2]
resultado = step_fun(A)
print(f"Resultado step: {resultado}")

#lineal y = m*x +b
def lineal_fun(inputs):

    m = 1 #pendiente
    b = 0 #sesgo
    
    output = []

    for i in inputs:
        output.append(m*i + b)
    
    return output

A = [0, 1, -1]
resultado_lineal = lineal_fun(A)
print(f"Resultado lineal: {resultado_lineal}")


#Sigmoide
def sigmoide(inputs):

    output = []

    for i in inputs:
        
        r = 1 / (1 + np.e**(-i))
        output.append(r)

    return output

A = [-100, 1, 80]
resultado_sigmoide = sigmoide(A)
print(f"Resultado sigmoide: {resultado_sigmoide}")

def Re_Lu_simplificada(inputs):

    output = []

    for i in inputs:
        if i > 0 :
            output.append(i)
        else:
            output.append(0)

    return output

Resultado_Re_Lu_Simp = Re_Lu_simplificada(A)
print(f"Resultado Re_Lu Simplicado: {Resultado_Re_Lu_Simp}")

class Activation_ReLu:

    def forward(self, inputs):

        #calculamos el output

        self.output = np.maximum(0, inputs)

inputs = [2, 0, -1.3, 5.3, 2.7, -3.1, 7.5, -100]

A1 = Activation_ReLu()

A1.forward(inputs)
R = Re_Lu_simplificada(inputs)

print(A1.output)
print(R)