import streamlit as st


if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The News is Fake')
    else :
        st.write('The News is Real')
