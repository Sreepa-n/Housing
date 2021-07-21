from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

import joblib
model = joblib.load('California_Model.pkl')
#model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        longitude = float(request.form['longitude'])
        latitude=float(request.form['latitude'])
        median_age=int(request.form['housing_median_age'])
        rooms = int(request.form['roomsperhousehold'])
        bedrooms = int(request.form['bedroomsperroom'])
        population = int(request.form['population_per_household'])
        households = int(request.form['households'])
        median_income = int(request.form['median_income'])
        ocean_proximity=request.form['ocean_proximity']
        if(ocean_proximity=='Island'):
            ocean_proximity_island=1
            ocean_proximity_nearbay=0
            ocean_proximity_nearocean=0
            ocean_proximity_inland=0
            ocean_proximity_ocean= 0
        elif(ocean_proximity=='Near Bay'):
            ocean_proximity_island = 0
            ocean_proximity_nearbay = 1
            ocean_proximity_nearocean = 0
            ocean_proximity_inland = 0
            ocean_proximity_ocean = 0
        elif(ocean_proximity=='Near Ocean'):
            ocean_proximity_island = 0
            ocean_proximity_nearbay = 0
            ocean_proximity_nearocean = 1
            ocean_proximity_inland = 0
            ocean_proximity_ocean = 0
        elif(ocean_proximity=='Inland'):
            ocean_proximity_island = 0
            ocean_proximity_nearbay = 0
            ocean_proximity_nearocean = 0
            ocean_proximity_inland = 1
            ocean_proximity_ocean = 0
        else:
            ocean_proximity_island = 0
            ocean_proximity_nearbay = 0
            ocean_proximity_nearocean = 0
            ocean_proximity_inland = 0
            ocean_proximity_ocean = 1


        prediction=model.predict([[longitude,latitude,median_age,households,median_income,rooms,bedrooms,population,ocean_proximity_island,ocean_proximity_nearbay,ocean_proximity_nearocean,ocean_proximity_inland,ocean_proximity_ocean]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry price not available")
        else:
            return render_template('index.html',prediction_text="Price of the house is {}".format(output))
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)