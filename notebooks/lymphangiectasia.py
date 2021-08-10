#meter en un dataframe cada una de las diferentes dolencias que se van a analizar con el programa
#En este caso el dataframe esta compuesta por la clase, el array de las imagenes y el fullpath de la imagen
# En este caso particular se ha realizado un data augmentation ya que se contaba con un numero bastante inferior de imagenes con respecto a las otras clases. 
import pandas as pd 
import numpy as np
import glob
import sys,os
import cv2
from os import listdir
from os.path import isfile, join
dir = os.path.dirname
path = dir(dir(os.path.abspath(__file__)))
print(path)
sys.path.append(path)

import SRC.utils_.mining_data_tb as mn


df_Lymphangiectasia = mn.importame(carpeta = "Lymphangiectasia",path=path)
print(df_Lymphangiectasia)

mn.aumentame(df_Lymphangiectasia)

df_Aumento = mn.importame2(carpeta = "data_augmentation_lymphangiectasia",clase="Lymphangiectasia",path=path)
mn.limpio_csv(df_Aumento,"data_augmentation")

