import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

## the first line of code to run
application = Flask(__name__)
app = application

### import ridge regressor and standard scaler pickle
## call the model 
ridge_model=pickle.load(open('models/ridge.pkl', "rb"))  ## refering to model folder
## call the standard scaler
standard_scaler=pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def hello_world():
    #return "<h1>Hello, World!</h1>"   ## was just for testing on localhost
    return render_template("index.html")  ## ref to html file in this folder

## for the prediction

@app.route("/predictdata",methods=["GET", "POST"])
def predict_datapoint():
    if request.method=="POST":
        #pass
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        ### scaling my new datapoints for predicting
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)  ## making the prediction

        return render_template('home.html',result=result[0])


    else:   ## equal to when GET or !==Post request
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")  ## mapped to my local ip address