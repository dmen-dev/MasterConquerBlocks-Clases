import numpy as np

class Loss:
    def calculate(self, output, y):
        # CALCULAR LA PÉRDIDA DE MUESTRA
        perdida_muestra = self.forward(output, y)
        # CALCULAR LA PÉRDIDA MEDIA
        perdida_media = np.mean(perdida_muestra)
        return perdida_media

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # CALCULO DEL NUMERO DE DATOS
        n_datos = len(y_pred)

        # LIMITAR LOS VALORES DE Y_PRED PARA EVITAR LOG(0)
        y_pred_limite = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # SOLO PARA VALORES CATEGORICOS
        if len(y_true.shape) == 1:  # ES IGUAL A UN VECTOR
            confianza = y_pred_limite[range(n_datos), y_true]
        elif len(y_true.shape) == 2:  # ES IGUAL A UNA MATRIZ
            confianza = np.sum(y_pred_limite * y_true, axis=1)

        # CALCULAR LA PÉRDIDA NEGATIVA
        Perdida_negativa = -np.log(confianza)

        return Perdida_negativa


