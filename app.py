import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] +''+ news_df['title']



ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-z A-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in
    stopwords.words('english')]
    stemmed_content =' '.join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)


X = news_df['content'].values
y = news_df['label'].values



vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train,X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train,Y_train)





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


#  Local URL: http://localhost:8501
#   Network URL: http://192.168.106.43:8501
