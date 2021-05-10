from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger #Swagger generate frontend UI path
 
app = Flask(__name__)  #app start with this point
Swagger(app)

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome Flask User'

@app.route('/predict', methods = ['Get'])
def predict_note_authentication():
    """
    Let's authenticate the bank note
    this is using docstrings for specifications.
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true

        - name: skewness
          in: query
          type: number
          required: true

        - name: curtosis
          in: query
          type: number
          required: true

        - name: entropy
          in: query
          type: number
          required: true

    responses:
        200:
            description: The o/p values
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    #pass inputs
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return "Prediction values is " + str(prediction)

@app.route('/predict_file', methods = ['POST'])
def predict_note_file():
    """
    Let's authenticate the bank note
    this is using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true

    responses:
        200:
            description: The o/p values
    """
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return "Prediction values for txt file is " +str(list(prediction))

if __name__ == '__main__':
    app.run(debug=True)