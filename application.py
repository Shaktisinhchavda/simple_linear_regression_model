import pickle
from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
 
application = Flask(__name__)
app = application

ridge_model = pickle.load(open("models/ridge.pickel","rb"))
scaler_model = pickle.load(open("models/scaler.pickel","rb"))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction",methods=["POST","GET"])
def prediction():
    if request.method=="POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        region = float(request.form.get("region"))

        new_data = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,region]])
        result = ridge_model.predict(new_data)
        
        return render_template("home.html",results=result[0])

    else:
        return render_template("home.html")
if __name__=="__main__":
    app.run(host="0.0.0.0")