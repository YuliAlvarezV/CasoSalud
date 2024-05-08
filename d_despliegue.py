import numpy as np
import pandas as pd
import cv2 ### para leer imagenes jpeg
### pip install opencv-python
import _funciones as fn#### funciones personalizadas, carga de imágenes
import tensorflow as tf
import openpyxl

import sys
sys.executable
sys.path

if __name__=="__main__":

    #### cargar datos ####
    path = 'C:\\Users\\pablo\\Analitica 3\\dbsalud\\despliegue\\'
    x, _, files= fn.img3data(path) #cargar datos de despliegue

    x=np.array(x) ##imagenes a predecir

    x=x.astype('float')######convertir para escalar
    x/=254######escalar datos


    files2= [name.rsplit('.', 1)[0] for name in files] ### eliminar extension a nombre de archivo

    modelo=tf.keras.models.load_model('C:\\Users\\pablo\\Analitica 3\\dbsalud\\salidas\\fc_model.h5') ### cargar modelo
    prob=modelo.predict(x)


    clas = []
    for p in prob:
        if p > 0.52:
            clas.append('Demented')
        elif p < 0.50:
            clas.append('No Demented')
        else:
            clas.append('No ident')

    res_dict={
        "paciente": files2,
        "clas": clas   
    }
    resultados=pd.DataFrame(res_dict)

    resultados.to_excel('C:\\Users\\pablo\\Analitica 3\\dbsalud\\salidas\\clasificados.xlsx', index=False)