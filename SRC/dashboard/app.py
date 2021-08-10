import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
import requests
import sys,os
from flask import Flask, request, render_template
import argparse
import json
import pymysql
from sqlalchemy import create_engine

from tensorflow import keras
import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image

from tensorflow import keras

from sklearn import model_selection
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential

from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from PIL import Image, ImageOps
pato = os.path.dirname
direccion=pato(pato(pato(__file__)))
sys.path.append(direccion)
from notebooks.df_final import *

from SRC.utils_.mining_data_tb import *
from SRC.utils_.visualization_tb import *
from SRC.utils_.models import *
from SRC.utils_.apis_tb import *
from SRC.utils_.sql_tb import *

st.set_page_config(layout="wide") 
menu = st.sidebar.selectbox('Menu:',
            options=["Bienvenido", "Visualización","Json API-Flask","Modelos SQL","Predicciones de los modelos","Información"])

st.title(' Programa de reconocimiento de lesiones del tracto digestivo')


if menu == 'Bienvenido':
    st.write("En este proyecto se pretende analizar la capacidad de los diferentes modelos de _machine learning_ para el reconocimiento de diferentes lesiones que se producen en el intestino grueso y delgado. Se han tenido en cuenta las siguientes lesiones:") 
    st.write("1. **Pólipos:** los pólipos son un tipo de tumor generalmente benigno que se produce, entre otros sitios, en el revestimiento de las paredes del tracto digestivo. Se han tenido en cuenta dos tipos de pólipos, causantes del 90% de los pólipos existentes.")
    st.write("          - **Adenomas:** es un tipo de pólipo bastante complicado de identificar a simple vista, ya que suelen ser planos y no suelen tener una apariencia muy diferente a la mucosa. Además aun cuando se detectan, son complicados de delimitarlos para proceder a su extirpación. A largo plazo, este tipo de pólipos causan el 85% de los cánceres colorrectales. Por lo tanto, su identificación es muy importante.")
    st.write("          - **Hiperplásicos:** pólipos mucho más comunes que los anteriores pero que en un mínimo porcentaje causan cánceres. Además según diferentes estudios médicos, su identificación es bastante sencilla ya que sobresalen, a veces tienen una estructura que les une a la mucosa y aparecen como colgando en las paredes del tracto digestivo. Debido a la facilidad de identificación también son muy fáciles de delimitar para proceder a su extirpación.")
    st.write("2. **Úlceras:** yagas causadas por el debilitamiento de la mucosa que recubre el tracto digestivo. Las úlceras y la inflamación del intestino grueso y de la parte final del íleon suelen ser indicativos de enfermedad inflamatoria intestinal (Enfermedad de Crohn y Enfermedad de Colitis Ulcerosa). Por lo tanto, su identificación es muy importante para poder descartar o bien determinar la existencia de estas enfermedades crónicas.")
    st.write("3. **Lesiones causadas por la enfermedad de La linfagiectasia intestinal:** enfermedad crónica del aparato linfático. La enfermedad consiste en una dilatación de los vasos linfáticos intestinales que origina un trastorno del drenaje linfático. Por lo tanto, las lesiones se presentan con la aparición de líquido linfático en el tracto digestivo.")
    st.write("A parte de estas dolencias, también se han tenido en cuenta imágenes de **mucosa sana** para que el modelo sepa identificar aquellas partes del tracto que no presentan ningún tipo de lesiones.")

if menu == "Visualización":
    st.markdown('### Predicciones diferentes tipos de lesiones')
    st.write ("tipo 0 = Adenoma")
    st.write ("tipo 1 = Hiperplástico")
    st.write ("tipo 2 = Úlceras")
    st.write ("tipo 3 = Linfagiectasia")
    st.write ("tipo 4 = Mucosa normal")
    submenu=st.sidebar.selectbox(label="Modelos:",
            options=["lesiones","cnn sencillo","Cnn complejo","VGG16","Resnet 50"])
    if submenu=="lesiones":
        st.markdown('### Diferentes tipos de lesiones')
        grafico1 = Image.open(direccion + os.sep + 'resources' + os.sep + 'sample.png')
        st.image (grafico1,use_column_width=True)
    if submenu=="cnn sencillo":
        grafico1 = Image.open(direccion + os.sep + 'resources' + os.sep + 'prediccionescnn2.png')
        st.image (grafico1,use_column_width=True)
    if submenu=="Cnn complejo":
        grafico1 = Image.open(direccion + os.sep + 'resources' + os.sep + 'prediccionescnn1.png')
        st.image (grafico1,use_column_width=True)
    if submenu=="VGG16":
        grafico1 = Image.open(direccion + os.sep + 'resources' + os.sep + 'prediccionesvgg16.png')
        st.image (grafico1,use_column_width=True)
    if submenu=="Resnet 50":
        grafico1 = Image.open(direccion + os.sep + 'resources' + os.sep + 'prediccionesresnet50.png')
        st.image (grafico1,use_column_width=True)
if menu == "Json API-Flask":
    r = requests.get("http://localhost:8080/give_me_id?token_id=R70423563").json()
    df = pd.DataFrame(r)
    st.write(df)
if menu == "Modelos SQL":
        st.markdown('### Selección del mejor modelo')
        df = pd.read_csv(direccion + os.sep + 'data' + os.sep + "union.csv",nrows=50)
        st.write(df)
if menu == "Predicciones de los modelos":
    submenu=st.sidebar.selectbox(label="Predicciones de los modelos:",
            options=["Predicciones","Pruébame"])
    if submenu=="Predicciones":
        st.markdown('### Datos predecidos por los diferentes modelos')
        df = pd.read_csv(direccion + os.sep + 'data' + os.sep + "prediciones_vs_real.csv",nrows=50)
        st.write(df)
    if submenu=="Pruébame":
        st.title("Upload + Classification Example")
        uploaded_file = st.file_uploader("Choose a JPG image", type="jpg")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("Classifying...")
            X = np.stack(np.array(image))
            model_path = direccion + os.sep + 'models' + os.sep + 'VGG16.h5'
            new_model = keras.models.load_model(model_path)
            
            smallimage = cv2.resize(X, (100, 300))
            pred = new_model.predict(preprocess_input(np.array(smallimage).reshape(1, 100, 100, 3)))
            if np.argmax(pred) == 0:
                st.write ("Lesión: Pólipo clase adenoma")
            if np.argmax(pred) == 1:
                st.write ("Lesión: Pólipo hiperplastico")    
            if np.argmax(pred) == 2:
                st.write ("Lesión: Úlcera") 
            if np.argmax(pred) == 3:
                st.write ("Lesión: Mucosa normal") 
            if np.argmax(pred) == 4:
                st.write ("Lesión: Lymfangiectasia") 
if menu == "Información":
    st.markdown("### Para más información:")
    st.write(" 1. [Linkedin](https://www.linkedin.com/in/pabloeduardomartinezpicazo/)")
    st.write(" 2. Correo electónico: pabloeduardo.martinezpicazo@gmail.com")