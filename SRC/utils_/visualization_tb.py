import pandas as pd
import numpy as np
import sys,os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def predicciones_vs_verdad(X_test,y_pred):
    plt.figure(figsize=(20, 10))
    
    for i in range(18):
        ax = plt.subplot(3, 6, i + 1)
        plt.imshow(X_test[i])
        plt.title(np.argmax(y_pred[i]))
        plt.axis("off")
def horitas (array):
    plt.pie(array,labels=["Búsqueda info","EDA","Creación funciones","Importar a main","Deep-Learning","Flask","Streamlit","Presentación"])
    plt.legend(title = "Horas invertidas:",bbox_to_anchor=(1,0))
    plt.show() 