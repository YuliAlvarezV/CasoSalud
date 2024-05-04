import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

####instalar paquete !pip install keras-tuner

import keras_tuner as kt
from tensorflow.keras.utils import to_categorical

### cargar bases_procesadas ####

x_train = joblib.load('salidas\\x_train.pkl')
y_train = joblib.load('salidas\\y_train.pkl')
x_test = joblib.load('salidas\\x_test.pkl')
y_test = joblib.load('salidas\\y_test.pkl')

############################################################
################ Preprocesamiento ##############
############################################################

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

##########################################################
################ Redes convolucionales ###################
##########################################################

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compilacion del modelo con categorical_crossentropy loss y Adam optimizer
cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC', 'f1_score'])

# entrenamiento del modelo y primeras metricas de desempeño
y_train_one_hot = to_categorical(y_train, 4)
y_test_one_hot = to_categorical(y_test, 4)
cnn_model.fit(x_train, y_train_one_hot, batch_size=100, epochs=10, validation_data=(x_test, y_test_one_hot))

cnn_model.summary()

#######probar una red con regulzarización L2
reg_strength = 0.001

###########Estrategias a usar: regilarization usar una a la vez para ver impacto
dropout_rate = 0.1  


cnn_model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model with binary cross-entropy loss and Adam optimizer
cnn_model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['AUC', 'f1_score'])

# Train the model for 10 epochs
cnn_model2.fit(x_train, y_train_one_hot, batch_size=100, epochs=10, validation_data=(x_test, y_test_one_hot))

#####################################################
###### afinar hiperparameter ########################
#####################################################

##### función con definicion de hiperparámetros a afinar
hp = kt.HyperParameters()

def build_model(hp):
    
    dropout_rate=hp.Float('DO', min_value=0.3, max_value= 0.6, step=0.05)
    reg_strength = hp.Float("rs", min_value=0.0005, max_value=0.001, step=0.0001)
    optimizer = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop']) ### en el contexto no se debería afinar
    activation_fn = hp.Choice('activation', ['tanh', 'leaky_relu', 'relu'])  # la función de activación
    ####hp.Int
    ####hp.Choice
    

    model= tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation=activation_fn, input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation=activation_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_strength)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
  
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)
   
    model.compile(
        optimizer=opt, loss="categorical_crossentropy", metrics=['AUC', 'f1_score'],
    )
    
    
    return model

tuner = kt.RandomSearch(
    hypermodel=build_model,
    hyperparameters=hp,
    tune_new_entries=False, ## solo evalúe los hiperparámetros configurados
    objective=kt.Objective("val_AUC", direction="max"),
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld", 
)

tuner.search(x_train, y_train_one_hot, epochs=5, validation_data=(x_test, y_test_one_hot), batch_size=100)

fc_best_model = tuner.get_best_models(num_models=1)[0]
tuner.results_summary()
fc_best_model.summary()

train_results = fc_best_model.evaluate(x_train, y_train_one_hot, verbose=2)
test_results = fc_best_model.evaluate(x_test, y_test_one_hot, verbose=2)

#################### Mejor redes ##############
fc_best_model.save('salidas\\fc_model.h5')
cnn_model.save('salidas\\cnn_model.h5')


cargar_modelo=tf.keras.models.load_model('salidas\\cnn_model.h5')
cargar_modelo.summary()