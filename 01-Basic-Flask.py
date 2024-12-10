from flask import Flask
#basic flask we application
#from flask import Flask, the actual syntax has no colour, after saving file as .py you will get colour, sublime text
#save into same folder as your model, scaler and json
#create flask app with app = Flask(__name__)
app = Flask(__name__)
#uses routing system to display info on page with decorators
#@app.route('/') - home page with /
@app.route('/')
#function to return def index():
# return '<h1 as heading 1> FLASK APP IS RUNNING<h1>'
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'
#finally to run this file
#if __name__ == '__mai__': app.run()
#save the file and open up command prompt, need to know where it is located
if __name__ == '__main__':
    app.run()

#open command line, terminal, activate your environment activate conda or env activate tf2gpu
#ensure you have installed flask pip install flask
#need to change directory where file is saved
#cd /python/tf2/deployment/ tab to autocomplete
#now, python my_flaskapp.py, autocomplete and hit enter
#get warning, that's okay, makes sense, not deployed yet to live webb, control+c quit and kill flask app
#local connection at port 5000, type it in, at that IP address, get heading with flask app, control+c to kill the server
#use proper flask app to return model new prediction