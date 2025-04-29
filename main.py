import streamlit as st

st.title('Fake News Detector')
input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_text = vector.transform([input_text])
    prediction = model.predict(input_text)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else :
        st.write('The News is Real')


