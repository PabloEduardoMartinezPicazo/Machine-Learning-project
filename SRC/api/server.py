

from flask import Flask, request, render_template
import os,sys
import argparse
import json
from flask import Flask
import pymysql
from sqlalchemy import create_engine
dir = os.path.dirname
src_path = dir(dir(dir(__file__)))
sys.path.append(src_path)

from SRC.utils_.apis_tb import read_json, return_json
from SRC.utils_.sql_tb import MySQL, upload_sql


app = Flask(__name__)

@app.route("/") 
def home():
    """ Default path """
    return "Proyecto Reconocimiento de Lesiones Intestinales"

@app.route('/give_me_id', methods=['GET']) # localhost:8080/give_me_id?token_id=R70423563
# Returns the json with the information of my df
def give_id():
    s = request.args['token_id']
    if s == "R70423563":
        path_csv = src_path + os.sep + "data" + os.sep + "csvlimpio.csv"
        return return_json(path_csv)
    else:
        return "Wrong password"

@app.route('/mysql', methods=['GET']) # localhost:8080/mysql?upload=yes
# This function inserts to mySQL the df
def mysql():
    x = request.args['upload']
    if x == "yes":
        settings_file = dir(dir(__file__)) + os.sep + "utils_" + os.sep + "bd_info.json" # Path to bd_info.json
        json_readed = read_json(settings_file) # Imported from apis_tb, read json
        route = src_path + os.sep + "data" + os.sep + "csvlimpioSQL_limpio.csv"
        upload_sql(route, json_readed)
        return "Successfully uploaded"
    else:
        return "Insert 'upload=yes' to insert the data in SQL"


def main():
    
    settings_file = dir(__file__) + os.sep + "jason.json"
    
    # Load json from file
    json_readed = read_json(fullpath=settings_file)

    DEBUG = json_readed["debug"]
    HOST = json_readed["host"]
    PORT_NUM = json_readed["port"] 

    app.run(debug=DEBUG, host=HOST, port=PORT_NUM)

parser = argparse.ArgumentParser()
parser.add_argument("-x", "--x", type=str, help="password") # Important! --> type

if __name__ == "__main__": 

    # To run in cmd : python C:\Users\xyang\OneDrive\Escritorio\ARCHIVOS\THEBRIDGE\Data-Science-Bootcamp-21\ENTREGABLES\00_POYECTO_ML\src\api\server.py -x "Pablo"
    args = vars(parser.parse_args())
    print(args.values())
    if args["x"] == "Pablo": 
        main()
    else:
        print("wrong password")