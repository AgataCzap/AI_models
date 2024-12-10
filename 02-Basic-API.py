from flask import Flask, request, jsonify
import numpy as np  
from tensorflow.keras.models import load_model
import joblib

### basic flask application combine it with prediction function and set it up with flask view
#send request with postma
#copy and paste return function
#add your imports such as tensorflow, joblib and numpy
#copy and return prediction function


#### THIS IS WHAT WE DO IN POSTMAN ###
# STEP 1: Create New Request
# STEP 2: Select POST
# STEP 3: Type correct URL (http://127.0.0.1:5000/api/flower)
# STEP 4: Select Body
# STEP 5: Select JSON
# STEP 6: Type or Paste in example json request
# STEP 7: Run 02-Basic-API.py to launch server and confirm the site is running
# Step 8: Run API request

def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']
    
    flower = [[s_len,s_wid,p_len,p_wid]]
    
    flower = scaler.transform(flower)
    
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    
    class_ind = model.predict_classes(flower)
    
    return classes[class_ind][0]

#create your flask app, homepage, create new route, load model and scaler

app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'

#remember to add entire file path, if same folder, then just filenames
#if located somewhere else, will need full file path
#use the load_model and joblib.load, same lines as in notebook
# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")

#create route view to accept new data and return an answer
#@app.route('/api/flower', methods = ['POST'])
#saying that this routing page accepts post request
@app.route('/api/flower', methods=['POST'])
#define function
#content = request.json, when you send something to the page, it will be send in this form and return json
#add request to imports, use jsonify to return the results, build into flask
#take the request and grab json from it, pass into prediction function
#results = return_prediction(model, scaler, sample_json = content)
#return jsonify(results), save changes
def predict_flower():

    content = request.json
    
    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run()

#open your terminal, python example.py, should see tensorflow uploading it, importing it
#open up your browser, and upon opening it, IP address, flask app is running, you need to download postman
#download postman app, available for all, nice graphical user interface for sending off json requests
#open it up, upon launching will request to register, can skip below, create new request to send off to flask API
#file - create new - request - give name flower prediction - can give descrip, pass flower measurements, get back class pred
#select a collection or create new collection, predcition models - save
#graphical interface for sending requests, post data to web application, choose post, command prompt, everything is running
#local connection, take a look at your sublime code, view or routing, runs at /api/flower and accepts post methods
#say http//IP address/api/flower/ input into URL, select what we will send, click body, select raw, select JSON
#send the json code in same format as expected, so {"sepal length": 5.1} and so on
#once done, click send, make sure that app is running, no typos, should see status 200 html ok, see the returned prediction below in
#postman, send json code for preferred method, postman is one of the easiest to test your model and flask app