import streamlit as st
import joblib  # to load the saved vectorizer and model

# Load the vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')  # make sure you have this file
model = joblib.load('model.pkl')            # make sure you have this file

st.title('📰 Fake News Detector')
input_text = st.text_input('Enter news article text:')

def prediction(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.error('🚨 The News is Fake!')
    else:
        st.success('✅ The News is Real!')


