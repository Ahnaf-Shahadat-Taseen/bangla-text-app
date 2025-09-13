# app.py (Definitive Version with Correct IP Address)

import os
import streamlit as st
import pandas as pd

# Forcefully Set Hadoop Home and System Properties
hadoop_home_path = r"C:\hadoop"
os.environ["HADOOP_HOME"] = hadoop_home_path
os.environ["PATH"] += f";{os.path.join(hadoop_home_path, 'bin')}"

# Set PYSPARK_PYTHON to your python executable
# Find this by running 'where python' in your cmd
python_executable_path = "C:\\Users\\USER\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" #<-- CHANGE THIS IF YOURS IS DIFFERENT
os.environ["PYSPARK_PYTHON"] = python_executable_path


from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import RegexTokenizer, HashingTF, MinHashLSHModel
import unicodedata
import string
import plotly.express as px

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

# --- Spark Session and Model Loading ---
@st.cache_resource
def load_spark_assets():
    """
    Starts Spark, loads data, and re-creates the feature vectors needed by the model.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "final_minhash_lsh_model")
    data_path = os.path.join(base_path, "clustered_app_data")
    
    # --- Final Fix: Correct IP address and add memory config ---
    spark = SparkSession.builder \
        .appName("WebAppLoaderFinal") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.extraJavaOptions", f"-Dhadoop.home.dir={hadoop_home_path}") \
        .getOrCreate()
        
    model = MinHashLSHModel.load(model_path)
    df_app_raw = pd.read_parquet(data_path)
    spark_df_app_raw = spark.createDataFrame(df_app_raw)
    
    # Re-create the 'features' column
    clean_text_udf = udf(clean_text, StringType())
    df_cleaned = spark_df_app_raw.withColumn("cleaned_text", clean_text_udf(col("article")))
    tokenizer = RegexTokenizer(inputCol="cleaned_text", outputCol="tokens", pattern="\\s+")
    df_tokenized = tokenizer.transform(df_cleaned)
    hashingTF = HashingTF(inputCol="tokens", outputCol="features", numFeatures=20000)
    df_vectorized = hashingTF.transform(df_tokenized)
    
    return spark, model, df_app_raw, df_vectorized

spark, model, df_app, spark_df_app_with_features = load_spark_assets()

# --- UI Layout ---
st.title("🔎 Bangla News Article Similarity Search Engine")
st.markdown("Enter a Bangla sentence or article below to find the most similar articles.")

st.header("Find Similar Articles")
query_text = st.text_area("Enter your text here:", height=150)
k_neighbors = st.slider("Number of similar articles to find:", 1, 10, 3)

if st.button("Search"):
    if query_text:
        with st.spinner("Processing your text and searching for similar articles..."):
            query_df = spark.createDataFrame([(1, query_text)], ["id", "text"])
            
            clean_text_udf = udf(clean_text, StringType())
            query_cleaned = query_df.withColumn("cleaned_text", clean_text_udf(col("text")))
            tokenizer = RegexTokenizer(inputCol="cleaned_text", outputCol="tokens", pattern="\\s+")
            query_tokenized = tokenizer.transform(query_cleaned)
            hashingTF = HashingTF(inputCol="tokens", outputCol="features", numFeatures=20000)
            query_vectorized = hashingTF.transform(query_tokenized)
            
            result_df = model.approxNearestNeighbors(spark_df_app_with_features, query_vectorized.head().features, k_neighbors)
            
            st.success(f"Found the top {k_neighbors} most similar articles!")
            results = result_df.collect()
            
            for i, row in enumerate(results):
                st.subheader(f"Result #{i+1} (Similarity Distance: {row.distCol:.4f})")
                st.write(f"**Category:** {row.category}")
                with st.expander("Click to read the full article"):
                    st.write(row.article)
                st.markdown("---")
    else:
        st.warning("Please enter some text to search.")

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