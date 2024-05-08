import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 

x_train /=254 
x_test /=254

##### cargar modelo  ######

modelo=tf.keras.models.load_model('salidas\\fc_model.h5')

####desempeño en evaluación para grupo 1 (tienen demencia) #######
prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en evaluacion")### conocer el comportamiento de las probabilidades para revisar threshold

threshold_dem=0.52

pred_test=(modelo.predict(x_test)>=threshold_dem).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demente', 'NoDemente'])
disp.plot()

### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_dem).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demente', 'NoDemente'])
disp.plot()

####desempeño en evaluación para grupo 1 (No tienen demencia) #######
########### ##############################################################

prob=modelo.predict(x_test)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en evaluacion")### conocer el comportamiento de las probabilidades para revisar threshold

threshold_no_dem=0.50

pred_test=(modelo.predict(x_test)>=threshold_no_dem).astype('int')
print(metrics.classification_report(y_test, pred_test))
cm=metrics.confusion_matrix(y_test,pred_test, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demencia', 'NoDemencia'])
disp.plot()

### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_no_dem).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm=metrics.confusion_matrix(y_train,pred_train, labels=[1,0])
disp=metrics.ConfusionMatrixDisplay(cm,display_labels=['Demencia', 'NoDemencia'])
disp.plot()

####### clasificación final ################

prob=modelo.predict(x_test)

clas=['dem' if prob >0.52 else 'No dem' if prob <0.50 else "No ident" for prob in prob]

clases, count =np.unique(clas, return_counts=True)

count*100/np.sum(count)