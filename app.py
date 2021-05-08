from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)  #app start with this point
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome Flask User'

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    #pass inputs
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "Prediction values is " + str(prediction)

@app.route('/predict_file', methods = ['POST'])
def predict_note_file():
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return "Prediction values for txt is " + str(list(prediction))


if __name__ == '__main__':
    app.run(debug=True)