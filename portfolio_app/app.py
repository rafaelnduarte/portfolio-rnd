# importing packages
import flask
from flask import Flask
import pandas as pd
import pickle
import xgboost
from xgboost import XGBRegressor

print(pickle.format_version)
# importing model and features
ml = XGBRegressor()
#with open('model/xgb_new.pkl', 'rb') as file:
#    modelo_simples = pickle.load(file)
ml.load_model('model/xgb_new.pkl') 
with open('model/features.names', 'rb') as file:
    features = pickle.load(file)


app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('airbnb.html')

    if flask.request.method == 'POST':
        user_inputs ={
            'Latitude': flask.request.form['latitude'],
            'Longitude': flask.request.form['longitude'],
            'Minimum Nights': flask.request.form['minimum_nights'],
            'Available Days In A Year': flask.request.form['availability_365'],
            'Borough': flask.request.form['neigh_group_encoded'],
            'Room Type': flask.request.form['room_encoded']
        }
        # input para dataframe (em branco)
        df = pd.DataFrame(index=[0], columns=features)
        df = df.fillna(value=0)
        # filling in the dataframe
        for i in user_inputs.items():
            df[i[0]] = i[1]
        # converting to numeric
        df = df.astype(float)

        df
        
        #making predictions
        y_pred = ml.predict(df)[0]
        print(y_pred)

        return flask.render_template('airbnb.html', price=(y_pred * 10))
        

if __name__ == ('__main__'):
    app.run()