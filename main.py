import streamlit as st

# Streamlit App
st.title('📰 Fake News Detector')
input_text = st.text_area('Enter News Article Text:', height=200)

def preprocess_input(text):
    text = stemming(text)
    text = vector.transform([text])
    return text

def prediction(input_text):
    processed_input = preprocess_input(input_text)
    pred = model.predict(processed_input)
    return pred[0]

if st.button('Predict'):
    if input_text.strip() == "":
        st.warning("Please enter some news text to check.")
    else:
        pred = prediction(input_text)
        if pred == 1:
            st.error('🚨 The News is Fake!')
        else:
            st.success('✅ The News is Real!')


