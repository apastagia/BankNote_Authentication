import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

def welcome():
    return 'Welcome to Streamlit'

def predict_note_authentication(variance, skewness, curtosis, entropy):
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
    #pass inputs
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    return (prediction)

def main():
    st.title("Bank Authentication")
    html_temp = """
    <div style = "background-color:tomato;padding:10px>
    <h2 style = "color:white;text-align:center;">Streamlit Bank Authentication ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Variance")
    skewness = st.text_input("Skewness")
    curtosis = st.text_input("Curtosis")
    entropy = st.text_input("Entropy")

    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success("The output is {}".format(result))

    if st.button("About"):
        st.text("Let's Learn")
        st.text("Build with Streamlit")

if __name__ == "__main__":
    main()