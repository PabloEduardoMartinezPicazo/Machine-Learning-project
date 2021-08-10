import os,sys
dir = os.path.dirname
path =  dir(dir(os.path.abspath(__file__)))
sys.path.append(path)
import SRC.utils_.models as md
import SRC.utils_.mining_data_tb as mn
from notebooks.df_final import df
df_sql = mn.quitar_array(df)
mn.limpio_csv(df_sql,"SQL")