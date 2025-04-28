# fake_news_detector_app.py

import streamlit as st
import pickle

# Set page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Load vectorizer and model
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'vectorizer.pkl' and 'model.pkl' are available.")
    st.stop()

# Title and description
st.title('📰 Fake News Detector')
st.markdown("""
Welcome to the Fake News Detector!  
Enter a news article below, and the app will predict if it's **Fake** or **Real**.
""")

# Input box
input_text = st.text_area('📝 Enter News Article Text Here:', height=200)

# Prediction function
def predict_news(text):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return prediction[0]

# Button and output
if st.button('Detect'):
    if input_text.strip() == "":
        st.warning('⚠️ Please enter some text to analyze.')
    else:
        result = predict_news(input_text)
        if result == 1:
            st.error('🚨 The News is **Fake**')
        else:
            st.success('✅ The News is **Real**')

# Footer
st.markdown("""
---
Made with ❤️ using Streamlit
""")

