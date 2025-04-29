import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# Initialize the stemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function for preprocessing text
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    stemmed_words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(stemmed_words)

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df = df.fillna('')
    df['content'] = df['author'] + ' ' + df['title']
    df['content'] = df['content'].apply(preprocess_text)
    return df

# Train the model
@st.cache_resource
def train_model(data):
    X = data['content']
    y = data['label']
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_vectorized = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer

# Load data and train model
data = load_data()
model, vectorizer = train_model(data)

