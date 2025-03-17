import numpy as np

class Loss:
    def calculate(self, output, y):
        # CALCULAR LA PÉRDIDA DE MUESTRA
        sample_losses = self.forward(output, y)
        # CALCULAR LA PÉRDIDA MEDIA
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # CALCULO DEL NUMERO DE DATOS
        n_datos = len(y_pred)

        # LIMITAR LOS VALORES DE Y_PRED PARA EVITAR LOG(0)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # SOLO PARA VALORES CATEGORICOS
        if len(y_true.shape) == 1:  # ES IGUAL A UN VECTOR
            correct_confidences = y_pred_clipped[range(n_datos), y_true]
        elif len(y_true.shape) == 2:  # ES IGUAL A UNA MATRIZ
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # CALCULAR LA PÉRDIDA NEGATIVA
        negative_log_likelihoods = -np.log(correct_confidences)

        return negative_log_likelihoods
    
    def backwards(self, dvalues, y_true):

        muestras = len(dvalues)

        etiquetas = len(dvalues[0])

        if len(y_true.shape) == 1: #si es una columna

            y_true = np.eye(etiquetas)[y_true] #eye sirve para crear una matriz identidad, todo zeros y diagonal unos, pero al poner y_true al final se indica la neurona que se activa

            #ejemplo eye si y = [0,2,1]
            # [[1,0,0],
            # [0,0,1],
            # [0,1,0]]
        
        #calculo el gradiente
        self.dinputs = -y_true / dvalues

        #normalizo el gradiente con el numero de muestras
        self.dinputs = self.dinputs / muestras
        