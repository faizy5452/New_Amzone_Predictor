import pandas as pd
import numpy as np
import nltk
import os
import ssl
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# =======================
# NLTK Setup (Streamlit Safe)
# =======================
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt", download_dir=nltk_data_path)

# =======================
# Load and preprocess data
# =======================
df = pd.read_csv("realistic_amazon_products.csv")
df.drop("id", axis=1, inplace=True)

stemmer = SnowballStemmer("english")

# Tokenize & Stem
def tokenize_stem(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens]

# Create joined stemmed string for each product
df["stemmed_tokens"] = df.apply(lambda row: " ".join(tokenize_stem(row["title"] + " " + row["description"])), axis=1)

# TF-IDF Vectorizer (no tokenizer here, we're using preprocessed strings)
tfidf_vectorizer = TfidfVectorizer()

# Similarity Calculation
def cosine_sim(text1, text2):
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Product Search Function
def search_product(query):
    stemmed_query = " ".join(tokenize_stem(query))
    df['similarity'] = df['stemmed_tokens'].apply(lambda x: cosine_sim(stemmed_query, x))
    res = df.sort_values(by=['similarity'], ascending=False).head(10)[['title', 'description', 'category']]
    return res

# =======================
# Streamlit UI
# =======================
try:
    img = Image.open('amazon_logo.png')
    st.image(img, width=600)
except FileNotFoundError:
    st.warning("Logo image not found.")

st.title("🔍 Amazon Product Search & Recommender")
query = st.text_input("Enter product name")
submit = st.button("Search")

if submit and query.strip() != "":
    result = search_product(query)
    st.write(result)
elif submit:
    st.warning("Please enter a product name to search.")
