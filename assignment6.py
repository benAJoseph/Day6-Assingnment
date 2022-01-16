import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os
from pathlib import Path

st.title('Tweet Sentiment Classification')

sentiment_model = joblib.load('naive_model.joblib')
vectorizer = joblib.load('CountVectorizer.joblib')
inp_text = st.text_area('Paste a covid Tweet to check the sentiment',height=200)

vectorised_text = vectorizer.transform([inp_text])
pred = ''

def senti_pred(inp_text):
    prediction = sentiment_model.predict(inp_text)
    if prediction == 0:
        pred = 'Positive'
    else:
        pred = 'Negative'
    return pred
if st.button('Submit'):
    st.write('The Tweet you entered is:',senti_pred(vectorised_text))