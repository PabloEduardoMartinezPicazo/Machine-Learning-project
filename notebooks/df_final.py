
# en este apartado he unido los diferentes dataframes en uno con todas las clases 

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

from notebooks.adenoma import df_Adenoma
from notebooks.hyperplastic import df_Hyperplastic
from notebooks.lymphangiectasia import df_Lymphangiectasia,df_Aumento
from notebooks.normal_clean_mucosa import df_Normal_clean_mucosa
from notebooks.ulcer import df_Ulcer
import SRC.utils_.mining_data_tb as mn
df = pd.concat([df_Adenoma,df_Hyperplastic,df_Lymphangiectasia,df_Normal_clean_mucosa,df_Ulcer,df_Aumento])

df = mn.limpieza(df)
mn.limpio_csv(df,"final")