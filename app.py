import pickle
import string
import streamlit as st 

import nltk
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
    
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Tokenize
    text = nltk.word_tokenize(text)
    
    # 3. Remove non-alphanumeric
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    # 4. Remove stopwords & punctuation
    text = y
    y = []
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stem word
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
                
    
    return " ".join(y)

# LOADING Files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('mdel.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter your Message")
if st.button('Predict'):

    # 1 Transform Text
    transform_sms = transform_text(input_sms)

    # 2 Vectorize Text
    vector_input = tfidf.transform([transform_sms])

    # 3 Predict
    result = model.predict(vector_input)[0]

    # 4 Deploy
    if result == 1:
        st.header('Spam')
    else:
        st.header("Not Spam")