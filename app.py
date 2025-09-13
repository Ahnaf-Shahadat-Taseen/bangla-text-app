# app.py (Final, Spark-Free Version for Streamlit Cloud)

import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import string
import plotly.express as px
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Bangla Text Similarity Search",
    page_icon="🔎",
    layout="wide"
)

# --- Text Cleaning Function ---
def clean_text(text):
    if text is None: return ""
    text = unicodedata.normalize('NFKC', text)
    bangla_punc = "।"
    translator = str.maketrans('', '', string.punctuation + string.ascii_letters + string.digits)
    text = text.translate(translator)
    text = text.replace(bangla_punc, "")
    return ' '.join(text.split())

# --- Load Data and Create Feature Vectors (Cached for performance) ---
@st.cache_resource
def load_assets():
    """
    Loads the data with Pandas and creates feature vectors using Scikit-learn.
    This is much faster and more memory-efficient than Spark for an app.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "clustered_app_data")
    
    # Load the data using pandas
    df_app = pd.read_parquet(data_path)
    
    # Clean the articles
    df_app['cleaned_text'] = df_app['article'].apply(clean_text)
    
    # Create feature vectors using the same logic as HashingTF
    # This vectorizer will be used for both the dataset and user queries
    vectorizer = HashingVectorizer(n_features=20000, alternate_sign=False)
    
    # Transform all articles into a matrix of feature vectors
    doc_vectors = vectorizer.fit_transform(df_app['cleaned_text'])
    
    return df_app, vectorizer, doc_vectors

df_app, vectorizer, doc_vectors = load_assets()

# --- UI Layout ---
st.title("🔎 Bangla News Article Similarity Search Engine")
st.markdown("Enter a Bangla sentence or article below to find the most similar news articles.")

st.header("Find Similar Articles")
query_text = st.text_area("Enter your text here:", height=150)
k_neighbors = st.slider("Number of similar articles to find:", 1, 10, 3)

if st.button("Search"):
    if query_text:
        with st.spinner("Processing your text and searching for similar articles..."):
            # 1. Clean the user's query
            cleaned_query = clean_text(query_text)
            
            # 2. Transform the query into a vector using the SAME vectorizer
            query_vector = vectorizer.transform([cleaned_query])
            
            # 3. Calculate cosine similarity between the query and all articles
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # 4. Get the indices of the top N most similar articles
            # We use argsort to get indices, then reverse and take the top N
            top_indices = np.argsort(similarities)[-k_neighbors:][::-1]

            # --- Display Results ---
            st.success(f"Found the top {k_neighbors} most similar articles!")
            
            for i, doc_index in enumerate(top_indices):
                # Get the result from the pandas DataFrame
                result = df_app.iloc[doc_index]
                similarity_score = similarities[doc_index]
                
                st.subheader(f"Result #{i+1} (Similarity Score: {similarity_score:.4f})")
                st.write(f"**Category:** {result['category']}")
                with st.expander("Click to read the full article"):
                    st.write(result['article'])
                st.markdown("---")
    else:
        st.warning("Please enter some text to search.")

# --- Data Exploration Section ---
st.header("Explore the Dataset")

st.subheader("Interactive Chart: Distribution of Articles by Cluster")
cluster_counts = df_app['cluster_id'].value_counts().reset_index()
cluster_counts.columns = ['Cluster ID', 'Number of Articles']
fig = px.bar(
    cluster_counts, x='Cluster ID', y='Number of Articles',
    title="Article Count per Cluster", text_auto=True
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Sample Data")
st.dataframe(df_app.head(10))
