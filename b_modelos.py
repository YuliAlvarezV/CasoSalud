import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
from sklearn import metrics ### para analizar modelo
from sklearn.ensemble import RandomForestClassifier  ### para analizar modelo
import pandas as pd
import tensorflow as tf

import cv2 ### para leer imagenes jpeg

from matplotlib import pyplot as plt #
from sklearn.preprocessing import label_binarize
from tensorflow.keras.utils import to_categorical


### cargar bases_preprocesadas ####

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') ## para poder escalarlo
x_train /=254 ### escalaro para que quede entre 0 y 1
x_test /=254

###### verificar tamaños

x_train.shape
x_test.shape


np.product(x_train[1].shape) ## cantidad de variables por imagen

np.unique(y_train, return_counts=True)
np.unique(y_test, return_counts=True)

##### convertir a 1 d array ############
x_train2=x_train.reshape(5121, 30000)
x_test2=x_test.reshape(1279, 30000)
x_train2.shape
x_test2.shape

############################################################
################ Probar modelos de tradicionales#########
############################################################

#################### RandomForest ##########

rf=RandomForestClassifier()
rf.fit(x_train2, y_train)

pred_train=rf.predict(x_train2)
print(metrics.classification_report(y_train, pred_train))
pred_test=rf.predict(x_test2)
print(metrics.classification_report(y_test, pred_test))

# Convertir las etiquetas de clase en un formato binario
#y_score_binary = label_binarize(pred_train, classes=[0, 1]) 
metrics.roc_auc_score(y_train, pred_train, average='macro', multi_class='ovr')
#y_score_binaryt = label_binarize(pred_test, classes=[0, 1]) 
metrics.roc_auc_score(y_test, pred_test, average='macro', multi_class='ovr')

############################################################
################ Probar modelos de redes neuronales #########
############################################################

fc_model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

##### configura el optimizador y la función para optimizar ##############

<<<<<<< HEAD
fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'f1_score', 'accuracy'])
=======
fc_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'f1_score'])
>>>>>>> e21224ab6075ced96d5d2f948a106cfa12d8a105

#####Entrenar el modelo usando el optimizador y arquitectura definidas #########

#y_train_one_hot = to_categorical(y_train, 3)
#y_test_one_hot = to_categorical(y_test, 3)
fc_model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))

#########Evaluar el modelo ####################

test_results = fc_model.evaluate(x_test, y_test, verbose=2)