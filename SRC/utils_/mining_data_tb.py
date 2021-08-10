#funciones de limpieza de los diferentes dataframes 
import pandas as pd 
import numpy as np
import glob
import sys,os
import cv2
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator


#función para crear el dataframe con las imagenes, sus clases y su fullpath.

def importame (carpeta,path):
    data_path = path + os.sep + "data" + os.sep + carpeta

    only_image_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    dict_ = []
    for image_name in only_image_names:
        if ".jpg" in image_name:
            image_fullpath = data_path + os.sep + image_name
            image_cv = cv2.imread(image_fullpath) 
            image_cv = cv2.resize(image_cv, (100, 100))  
            dict_.append({"Lesiones":image_cv, "tipos":carpeta, "Fullpath":image_fullpath})
        else:
            print(image_name)
    df_carpeta = pd.DataFrame(dict_)
    
    return df_carpeta


#función para crear el dataframe de las imagenes aumentadas -> tiene un argumento más porque el nombre del archivo y la clase es diferente.

def importame2 (carpeta,clase,path):
    data_path = path + os.sep + "data" + os.sep + carpeta

    only_image_names = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    dict_ = []
    for image_name in only_image_names:
        if ".jpg" in image_name:
            image_fullpath = data_path + os.sep + image_name
            image_cv = cv2.imread(image_fullpath) 
            image_cv = cv2.resize(image_cv, (100, 100))  
            dict_.append({"Lesiones":image_cv, "tipos":clase, "Fullpath":image_fullpath})
        else:
            print(image_name)
    df_carpeta = pd.DataFrame(dict_)
    
    return df_carpeta

#data augmentation ->usada solo para las imágenes de Lymphangiectasia.

def aumentame (df,no_me_llames_bb=False,fotos=1000):
    if no_me_llames_bb==True:
        df_lymphagiectasia1 = np.stack(np.array(df["Lesiones"]))
        df_lymphagiectasia1 = df_lymphagiectasia1.reshape(592,256,256,1)
        datagen  = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0.2,2.0],rescale=1./255)

        i = 0
        for batch in datagen.flow(df_lymphagiectasia1, batch_size=1,
                                        save_to_dir='data_augmentation_lymphangiectasia', save_format='jpg'):
            i += 1
            if i > fotos:
                break

#mínima limpieza y labeling del dataframe final para poder trabajar con los diferentes modelos

def limpieza (df):
    df.reset_index(drop=True,inplace=True)
    df["tipos"] = df["tipos"].map({"Adenoma":0,"Hyperplastic":1,"Ulcer":2,"Normal_clean_mucosa":3,"Lymphangiectasia":4})
    return(df)

#limpieza dataset, añadir dimensiones y quitar la columnar de los arrays para subirlo a SQL

def quitar_array(df):
    df_sql = df.drop(columns=["Lesiones"])
    df_sql["dimensiones"] = "100x100"
    return df_sql

#guardar el csv 
def limpio_csv(df,lesion):
    dirs = os.path.dirname
    return df.to_csv(dirs(dirs(dirs(__file__))) + os.sep + "data" + os.sep + (f"csvlimpio{lesion}.csv"),index=False)

def predicciones_vs_real(pred1,pred2,pred3,pred4,real):
    predicciones1 = []
    for i in range(len(real)):
        a = np.argmax(pred1[i])
        predicciones1.append(a)
    predicciones2 = []
    for i in range(len(real)):
        a = np.argmax(pred2[i])
        predicciones2.append(a)
    predicciones3 = []
    for i in range(len(real)):
        a = np.argmax(pred3[i])
        predicciones3.append(a)
    predicciones4 = []
    for i in range(len(real)):
        a = np.argmax(pred4[i])
        predicciones4.append(a)
    
    predicciones_vs_real = pd.DataFrame([predicciones1,predicciones2,predicciones3,predicciones4,real]).transpose()
    predicciones_vs_real.columns = ['cnn2', 'cnn1', 'vgg16', "resnet50",'test']
    return predicciones_vs_real

def union_sql (df1,df2,df3,df4):
    union_ = pd.concat([df1,df2,df3,df4])
    return union_

def reshapeo (X_train,X_test):
    X_train1 = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
    X_test1 = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3]))
    return X_test1,X_train1


