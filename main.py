import streamlit sa st

st.title("📰 Fake News Detection")
st.write("Enter the news article below to check if it's Real or Fake.")

user_input = st.text_area("News Article Text", height=200)

if st.button("Predict"):
    if user_input:
        processed_input = preprocess_text(user_input)
        vectorized_input = vectorizer.transform([processed_input])
        prediction = model.predict(vectorized_input)[0]
        result = "Real News 🟢" if prediction == 1 else "Fake News 🔴"
        st.subheader("Prediction:")
        st.success(result)
    else:
        st.warning("Please enter the news article text to make a prediction.")


