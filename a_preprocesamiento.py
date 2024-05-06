import numpy as np
import cv2 ### para leer imagenes jpeg
from matplotlib import pyplot as plt ## para gráfciar imágnes
import joblib ### para descargar array
import _funciones as fn#### funciones personalizadas, carga de imágenes

############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

img1=cv2.imread('C:\\Users\\pablo\\Analitica 3\\CasoSalud\\Alzheimer_Dataset\\test\\MildDemented\\26 (19).jpg')
img2 = cv2.imread('C:\\Users\\pablo\\Analitica 3\\CasoSalud\\Alzheimer_Dataset\\test\\ModerateDemented\\27 (2).jpg')

############################################
##### ver ejemplo de imágenes cargadas ######
#############################################

plt.imshow(img1)
plt.title('MildDemented')
plt.show()

plt.imshow(img2)
plt.title('ModerateDemented')
plt.show()

###### representación numérica de imágenes ####

img2.shape ### tamaño de imágenes
img1.shape
img1.max() ### máximo valor de intensidad en un pixel
img1.min() ### mínimo valor de intensidad en un pixel

np.prod(img1.shape) ### mas de 100 mil observaciones cada imágen

#### dado que se necesitarían muchas observaciones (imágenes para entrenar)
#### un modelo con tantas observaciones y no tenemos, vamos a reescalar las imágenes

img1_r = cv2.resize(img1 ,(100,100))
plt.imshow(img1_r)
plt.title('MildDemented')
plt.show()
np.prod(img1_r.shape)

img2_r = cv2.resize(img2 ,(100,100))
plt.imshow(img2_r)
plt.title('ModerateDemented')
plt.show()
np.prod(img2_r.shape)

################################################################
######## Código para cargar todas las imágenes #############
####### reducir su tamaño y convertir en array ################
################################################################


width = 100 #tamaño para reescalar imágen
num_classes = 2 #clases variable respuesta
trainpath = 'Alzheimer_Dataset/train/'
testpath = 'Alzheimer_Dataset/test/'

x_train, y_train= fn.img2data(trainpath) #Run in train
x_test, y_test = fn.img2data(testpath) #Run in test

plt.imshow(x_test[0])
plt.title('MildDemented')
plt.show()

#### convertir salidas a numpy array ####
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train.shape

np.prod(x_train[1].shape)
y_train.shape


x_test.shape
y_test.shape

####### salidas del preprocesamiento bases listas ######

joblib.dump(x_train, "salidas\\x_train.pkl")
joblib.dump(y_train, "salidas\\y_train.pkl")
joblib.dump(x_test, "salidas\\x_test.pkl")
joblib.dump(y_test, "salidas\\y_test.pkl")