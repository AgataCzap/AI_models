from flask import Flask, render_template, session, redirect, url_for, session #also import redirect
from flask_wtf import FlaskForm #import flask form, inherit form class
from wtforms import TextField,SubmitField
from wtforms.validators import NumberRange

#deployment aspect of connecting model to HTML website, any user can use the model by interacting it
#HTML and Jinja templates, what this requires, application, add few things to it to connect to template folder
#template folder has several html files to format the website
#for the application is to connect it to html files, use flask to create an html form and inject into home.html
#use flask to accept submitteed html form data
#use flask to return prediction to prediction.html
#submit on home.html file that uses a flask based form to accept user inpurt, sends this input back to flask app
#prediction.html file returns prediction once prediction function stops running
#use the files as reference, extra space can break connections here, any type will f it up
#previous app files show just simple flask page that sends json and processes it through the model
#not actual html pages, connect to html files, don't need request or jsonify, feed from html form instead
#from flask import render_template, renders template, need session to grab session data, URL_for that grabs different URLs
#simple lib to create html forms in flask, from wtforms import TextField, SubmitField, textfield is 4 empty boxes for user to type in
#submit is for the user to submit

import numpy as np  
from tensorflow.keras.models import load_model
import joblib



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



app = Flask(__name__)
# Configure a secret SECRET_KEY, to accept html forms correctly
# We will later learn much better ways to do this!!
#app.config['SECRET_KEY'] = string, allows forms to work, prevents hacking, ensures the current user can go to the next page
app.config['SECRET_KEY'] = 'mysecretkey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
flower_model = load_model("final_iris_model.h5")
flower_scaler = joblib.load("iris_scaler.pkl")


# Now create a WTForm Class
# Lots of fields available:
# http://wtforms.readthedocs.io/en/stable/fields.html
#can be saved into a different pyfile
#class FlowerForm(FlaskForm): basic form class that is inbuilt in wtf library
#create textfields with label sepal length, create text fields for all features
#sep_len = TextField('Sepal Length')
class FlowerForm(FlaskForm):
    sep_len = TextField('Sepal Length')
    sep_wid = TextField('Sepal Width')
    pet_len = TextField('Petal Length')
    pet_wid = TextField('Petal Width')
#using flask form library to create text fields, user will see them
    #finish off so create submit button submit = SubmitField('Analyze') can also be empty
    submit = SubmitField('Analyze')


#create home page, build index

@app.route('/', methods=['GET', 'POST']) #pass both get and post methods, on this page
def index():

    # Create instance of the form.
    form = FlowerForm()
    # If the form is valid on submission (we'll talk about validation next)
    #if form.validate_on_submit(): ensures text is submitted
    if form.validate_on_submit():
        # Grab the data from the breed on the form., saves things into current session
        #session['sep_len'] = form.sep_len.data, grab all features from the form, get data for all

        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data
#to make sure this form works we need to ensure that the methods on this actual page must be both get and post
        #go back and forth between the form
        #once user submits info, return redireect(url_for("prediction")) redirect to prediction site
        return redirect(url_for("prediction"))

#have to show something before submission, which means add inline with if
    #return render_template('home.html', form=form), that's it for homepage view, new flower form, 4 text boxes and submit
    #once user hits submit button, grab data from the form, from session and redirects to predictions page, string code
    return render_template('home.html', form=form)

    #change the name of function to prediction instead of flower, also change namee for prediction page
@app.route('/prediction') #change to prediction here
#prediction page pretty simple, content is now empty dict
def prediction():

    content = {} #empty dict
#so now grab data, initially a form so convert to float, grab from current session
    #content['sepal_length'] = float(session['sep_len']), for all features
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])
#key codes must sample sample json for the model
    #empty content dict, set up the content dictionary for the prediction function
    #results = return_prediction(model, scaler, sample_json = content dict created)
    results = return_prediction(model=flower_model,scaler=flower_scaler,sample_json=content)
#return render_template('prediction.html', results=results), get results of the function, can be used within pred html
    return render_template('prediction.html',results=results)
#connect prediction html to pred function, and home route to the form, prepared for you, how to connect to each html
#open files from the templates folder

if __name__ == '__main__':
    app.run(debug=True)