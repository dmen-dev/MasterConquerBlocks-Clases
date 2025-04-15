from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  #Modelo más utilizado, por ejemplo, utilizado para predecir un precio de pisos cuando tenemos un modelo entrenado para ello
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

#Generar un dataset sintetico
X, Y = make_classification(n_samples = 1000, n_features = 20, n_classes = 2, random_state = 3)
 

#Dividir el dataset en train y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 5)

#Creamos un modelo predictivo
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)

#Predicciones del modelo
Y_probs = model.predict_proba(X_test)[:,1]

#Calcular el area AUC_ROC

auc_roc = roc_auc_score(Y_test, Y_probs)
print(f"AUC-ROC: {auc_roc}")

fpr, tpr, thresholds = roc_curve(Y_test, Y_probs)


#Grafico las curvas ROC

plt.figure(figsize = (8,6))
plt.plot(fpr, tpr, color = 'orange', label = f'ROC curve (area= {auc_roc:.2f})')
plt.plot([0,1],[0,1], color = 'darkgrey', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc = "lower right")
plt.show()

