import streamlit as st
import numpy as np
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')

# Load dataset
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Preprocessing function
ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)

# Splitting data
X = news_df['content'].values
y = news_df['label'].values

vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

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

