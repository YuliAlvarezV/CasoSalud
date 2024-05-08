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
    path = 'C:\Users\GOMEZ\Documents\2024-1\ANALITICA\CasoSalud\\dbsalud\\despliegue\\'
    x, _, files= fn.img3data(path) #cargar datos de despliegue

    x=np.array(x) ##imagenes a predecir

    x=x.astype('float')######convertir para escalar
    x/=254######escalar datos


    files2= [name.rsplit('.', 1)[0] for name in files] ### eliminar extension a nombre de archivo

    modelo=tf.keras.models.load_model('C:\Users\GOMEZ\Documents\2024-1\ANALITICA\CasoSalud\dbsalud\\salidas\\fc_model.h5') ### cargar modelo
    prob=modelo.predict(x)


    clas=['Demented' if prob >0.52 else 'No Demented' if prob <0.50 else "No ident" for prob in prob]

    res_dict={
        "paciente": files2,
        "clas": clas   
    }
    resultados=pd.DataFrame(res_dict)

    resultados.to_excel('C:\Users\GOMEZ\Documents\2024-1\ANALITICA\CasoSalud\\dbsalud\\salidas\\clasificados.xlsx', index=False)