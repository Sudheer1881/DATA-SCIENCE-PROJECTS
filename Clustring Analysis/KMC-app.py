import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image  


# Set the page configuration
st.set_page_config(page_title="K-Means Clustering", layout="wide")


import base64

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Define the background image path
bg_image_path = r"C:\Users\HP\Downloads\360_F_1138316894_IgAPt8MxWT9KL503qi77xvU1088ue9WZ.jpg" # Change this to your image path

# Convert the image to base64
base64_img = get_base64_image(bg_image_path)

# Inject the CSS with the base64 image
page_bg_img = f"""
<style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
</style>
"""

# Apply the CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# Load the trained K-Means model
with open(r"C:\Users\HP\OneDrive\Desktop\Clustering Analysis\KMC.pkl", 'rb') as file:
    kmeans = pickle.load(file)

FEATURE_NAMES = [
    'Birth Rate', 'Business Tax Rate', 'CO2 Emissions',
    'Days to Start Business', 'Energy Usage', 'GDP', 'Health Exp % GDP',
    'Health Exp/Capita', 'Hours to do Tax', 'Infant Mortality Rate',
    'Internet Usage', 'Lending Interest', 'Life Expectancy Female',
    'Life Expectancy Male', 'Mobile Phone Usage', 'Population 0-14',
    'Population 15-64', 'Population 65+', 'Population Total',
    'Population Urban', 'Tourism Inbound', 'Tourism Outbound'
]





# Sidebar
st.sidebar.header("Enter Values for Numerical Columns")
user_input = [st.sidebar.number_input(feature, min_value=0, max_value=20000, step=5) for feature in FEATURE_NAMES]
input_data = np.array(user_input).reshape(1, -1)


# Main page layout
st.markdown(
    """
    <div style="
        background-color: #2C3E50;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.3);
    ">
        <h1 style="color: white; font-size: 32px; font-weight: bold;">🚀 Cluster Analysis</h1>
    </div>
    """, 
    unsafe_allow_html=True
)




import time


# Predict button (Top Centered)
st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
if st.button("Predict 🚀"):
    if input_data.shape[1] != kmeans.n_features_in_:
        st.error(f"⚠️ Expected {kmeans.n_features_in_} features, but got {input_data.shape[1]}.")
    else:
        cluster = kmeans.predict(input_data)[0]

        # Progress Bar Effect
        progress_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent + 1)

        # Success Message with Enhanced Styling
        st.markdown(
            f"""
            <div style='background-color: #222831; padding: 15px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white; font-size: 28px; font-weight: bold;'>🎯 Predicted Cluster: {cluster}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confetti Effect
        st.snow()

        
        


# Layout Setup (First Row: Upload & Data Summary)
col1, col2 = st.columns(2)
with col1:
    st.title("Upload Trained Data")
    uploaded_file = st.file_uploader("Upload File", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head())
with col2:
    st.title("Data Summary")
    if uploaded_file is not None:
        st.write(df.describe())

# Second Row: Histogram & Box Plot
col3, col4 = st.columns(2)
with col3:
    if uploaded_file is not None:
        st.title("Histogram")
        column = st.selectbox("Select Column for Histogram:", df.columns)
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=20, edgecolor='black')
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)
with col4:
    if uploaded_file is not None:
        st.title("Box Plot")
        column = st.selectbox("Select Column for Box Plot:", df.columns, key="box")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_xlabel(column)
        ax.set_title(f"Box Plot of {column}")
        st.pyplot(fig)

# Third Row: Scatter Plot & Cluster Heatmap
col5, col6 = st.columns(2)
with col5:
    if uploaded_file is not None:
        st.title("Scatter Plot")
        col_x = st.selectbox("Select X-axis for Scatter:", df.columns, key="scatter_x")
        col_y = st.selectbox("Select Y-axis for Scatter:", df.columns, key="scatter_y")
        fig, ax = plt.subplots()
        ax.scatter(df[col_x], df[col_y], c='blue', alpha=0.5)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")
        st.pyplot(fig)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)  # Ensure df is defined
    st.title("Correlation Heatmap")
    
    # Select only numeric columns before correlation calculation
    numerical_data = df.select_dtypes(include=[np.number])

    # Check if there are at least two numeric columns for correlation
    if numerical_data.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns to generate a heatmap.")
else:
    st.warning("Please upload a dataset to generate visualizations.")

