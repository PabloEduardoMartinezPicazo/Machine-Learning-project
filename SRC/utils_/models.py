#diferentes funciones para todo la creación de los diferentes modelos que se van a usar durante el proyecto.

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets, layers, models
from tensorflow import keras
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn import model_selection
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import pickle
import os,sys
from sklearn import metrics
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn import svm
from sklearn.metrics import recall_score
#Función para partir los datos en train y test

def particion (df,columnay,columnax):
    y = np.array(df[columnay])
    X = np.stack(np.array((df[columnax])))
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test
    
#Función para normalizar los datos 

def normalizacion (X_train,X_test):
    X_train = X_train/255
    X_test = X_test/255
    return (X_train,X_test)
#En es ta función se incluyen todos los modelos que se van a usar durante el proyecto. El modelo final seleccionado es el de cnn1 que es el que mejores resultados ha dado.
def modelaje (modelo,X_train,Y_train):
    """se tiene que poner cnn1,cnn2, vgg16 o resnet50 """
    if modelo == "cnn1":
        cnn1 = keras.Sequential([
        keras.layers.Conv2D(filters=8,  
                            kernel_size=(3, 3), 
                            input_shape=(100, 100, 3), 
                            padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2), 
                                padding="same"),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(), 
        keras.layers.Dense(32, activation='relu'), 
        keras.layers.Dense(5, activation="softmax") 
    ])
        cnn1.summary()
        cnn1.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        cnn1.fit(X_train, Y_train,
                            batch_size=64,
                            epochs = 10,
                            verbose=1,validation_split=0.2)
        return cnn1
    elif modelo == "cnn2":
        cnn2 = keras.Sequential([
        keras.layers.Conv2D(filters=8,  
                            kernel_size=(3, 3), 
                            input_shape=(100, 100, 3), 
                            padding='same'),
        keras.layers.Flatten(),  
        keras.layers.Dense(5, activation="softmax") 
    ])
        cnn2.summary()
        cnn2.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        
        cnn2.fit(X_train, Y_train,
                            batch_size=64,
                            epochs = 10,
                            verbose=1,validation_split=0.2)
        return cnn2
    elif modelo == "vgg16":
        base_model = VGG16(input_shape = (100, 100, 3),
                include_top=False,
                weights = 'imagenet')

        for layer in base_model.layers:
            layer.trainable = False

        x0 = layers.Flatten()(base_model.output)

        x1 = layers.Dense(10, activation='relu')(x0)

        x2 = layers.Dropout(0.5)(x1)

        x3 = layers.Dense(5, activation='softmax')(x2)

        vgg16 = tf.keras.models.Model(base_model.input, x3)

        vgg16.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])
        vgg16.summary()
        vgg16.fit(x=X_train,y=Y_train,
                    epochs = 5,
                    verbose=1)
        return vgg16
    elif modelo == "svc":  
        clf = svm.LinearSVC(max_iter=3000)
        clf.fit(X_train, Y_train)
        return clf
    elif modelo == "resnet50":
        base_model = ResNet50V2(input_shape=(100,100,3),include_top=False,weights = "imagenet",classifier_activation="softmax")
        for layer in base_model.layers:
            layer.trainable = False

        x0 = layers.Flatten()(base_model.output)

        x1 = layers.Dense(10, activation='relu')(x0)

        x2 = layers.Dropout(0.5)(x1)

        x3 = layers.Dense(5, activation='softmax')(x2)

        resnet50 = tf.keras.models.Model(base_model.input, x3)
        resnet50.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
        resnet50.summary()
        resnet50.fit(x=X_train,y=Y_train,
                epochs = 5,
                verbose=1)
        return resnet50


def prediccion_guardar(modelo,file_name,X_train,Y_train,X_test,Y_test,no_me_llames=False):
    if no_me_llames == True:
        modelo.save(f"../models/{file_name}.h5")
        
    y_pred = modelo.predict(X_test)
    predicciones = []
    for i in range(len(y_pred)):
        a = np.argmax(y_pred[i])
        predicciones.append(a)
    pred = {"predicciones":predicciones,"Valor real":Y_test}
    predVSreal  = pd.DataFrame(pred)
    return(y_pred,predVSreal)

#sacar los datos estadisticos del modelo.

def estadisticas (modelo,variable_filtro,X_train,Y_train,X_test,Y_test,y_pred):
    """se tiene que poner cnn1,cnn2, vgg16 o resnet50 """
    if variable_filtro == "cnn1":
        predicciones = []
        for i in range(len(y_pred)):
            a = np.argmax(y_pred[i])
            predicciones.append(a)

        test_acc = modelo.evaluate(X_test, Y_test)

        predicciones_scores1 = {"model":[variable_filtro],"parameters": "([keras.layers.Conv2D(filters=8,  kernel_size=(3, 3), input_shape=(100, 100, 3),  padding='same'),keras.layers.MaxPooling2D(pool_size=(2, 2),  padding=same),keras.layers.Dropout(0.25),keras.layers.Flatten(),  keras.layers.Dense(32, activation='relu'), keras.layers.Dense(5, activation=softmax)])","recall":"0.5544","score":[test_acc[1]]}
        Test_random_score_cnn1  = pd.DataFrame(predicciones_scores1)
        return Test_random_score_cnn1
    elif variable_filtro == "cnn2":
    
        predicciones = []
        for i in range(len(y_pred)):
            a = np.argmax(y_pred[i])
            predicciones.append(a)

        test_acc = modelo.evaluate(X_test, Y_test)
        
        
        
        predicciones_scores = {"model":[variable_filtro],"parameters":"keras.layers.Conv2D(filters=8, kernel_size=(3, 3), input_shape=(100, 100, 3),padding='same',keras.layers.Flatten(), keras.layers.Dense(5, activation=softmax)","recall":"0.5548","score":[test_acc[1]]}
        Test_random_score_cnn2  = pd.DataFrame(predicciones_scores)
        return Test_random_score_cnn2
    elif variable_filtro == "vgg16":
        predicciones = []
        for i in range(len(y_pred)):
            a = np.argmax(y_pred[i])
            predicciones.append(a)
        test_acc = modelo.evaluate(X_test, Y_test)
        
        predicciones_scores = {"model":[variable_filtro],"parameters":" x0 = layers.Flatten()(base_model.output) x1 = layers.Dense(10, activation='relu')(x0) x2 = layers.Dropout(0.5)(x1) x3 = layers.Dense(5, activation='softmax')(x2) vgg16 = tf.keras.models.Model(base_model.input, x3) vgg16.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])","recall":"0.5525","score":[test_acc[1]]}
        Test_random_score_vgg16  = pd.DataFrame(predicciones_scores)
        return Test_random_score_vgg16
    elif variable_filtro == "resnet50":
        predicciones = []
        for i in range(len(y_pred)):
            a = np.argmax(y_pred[i])
            predicciones.append(a)

        test_acc = modelo.evaluate(X_test, Y_test)
        predicciones_scores = {"model":[variable_filtro],"parameters":" x0 = layers.Flatten()(base_model.output) x1 = layers.Dense(10, activation='relu')(x0) x2 = layers.Dropout(0.5)(x1) x3 = layers.Dense(5, activation='softmax')(x2) resnet50 = tf.keras.models.Model(base_model.input, x3) resnet50.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])","recall":"0.5684","score":[test_acc[1]]}
        Test_random_score_resnet50  = pd.DataFrame(predicciones_scores)
        return Test_random_score_resnet50


