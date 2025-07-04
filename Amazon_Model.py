import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Download required NLTK resources
import nltk
import os
import ssl

# Create a safe nltk_data directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Ensure Streamlit knows this path
nltk.data.path.append(nltk_data_dir)

# Attempt to download punkt if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt', download_dir=nltk_data_dir)

# Tell nltk where to find the data
nltk.data.path.append(nltk_data_dir)



# Load and preprocess data
df = pd.read_csv("realistic_amazon_products.csv")
df.drop("id", axis=1, inplace=True)

stemmer = SnowballStemmer("english")

def tokenize_stem(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens]


df["stemmed_tokens"] = df.apply(lambda row: " ".join(tokenize_stem(row["title"] + " " + row["description"])), axis=1)


# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(text1, text2):
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def search_product(query):
    stemmed_query = tokenize_stem(query)
    df['similarity'] = df['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    res = df.sort_values(by=['similarity'], ascending=False).head(10)[['title', 'description', 'category']]
    return res

# Streamlit UI
try:
    img = Image.open('amazon_logo.png')
    st.image(img, width=600)
except FileNotFoundError:
    st.warning("Logo image not found.")

st.title("Search Engine and Product Recommendation")
query = st.text_input("Enter product name")
submit = st.button("Search")

if submit:
    res = search_product(query)
    st.write(res)
