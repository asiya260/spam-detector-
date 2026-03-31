# sms_lasso_streamlit.py
# Streamlit app for SMS Spam Classification Feature Selection using Lasso

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso

# -------------------------------
# Streamlit app title
# -------------------------------
st.title("SMS Spam Feature Selection with Lasso Regression")
st.write("Upload your SMS spam dataset and explore how Lasso selects important features.")

# -------------------------------
# Step 1: Upload dataset
# -------------------------------
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    st.subheader("Dataset Sample")
    st.dataframe(df.head())
    
    # -------------------------------
    # Step 2: TF-IDF Vectorization
    # -------------------------------
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['message'])
    y = df['label'].values
    total_features = X.shape[1]
    
    st.write(f"Total number of TF-IDF features: {total_features}")
    
    # Convert to dense array for Lasso
    X_dense = X.toarray()
    
    # -------------------------------
    # Step 3: Alpha selection
    # -------------------------------
    st.subheader("Select Lasso Alpha Values")
    alphas_input = st.text_input(
        "Enter alpha values separated by commas (e.g., 0.01,0.1,1)",
        "0.01,0.1,1"
    )
    
    # Parse alphas
    try:
        alphas = [float(a.strip()) for a in alphas_input.split(',')]
    except:
        st.error("Invalid alpha values. Please enter numbers separated by commas.")
        alphas = []
    
    # -------------------------------
    # Step 4: Apply Lasso
    # -------------------------------
    if st.button("Run Lasso Feature Selection") and alphas:
        st.subheader("Lasso Feature Selection Results")
        results = []
        for alpha in alphas:
            lasso = Lasso(alpha=alpha, max_iter=10000)
            lasso.fit(X_dense, y)
            coefficients = lasso.coef_
            non_zero = np.sum(coefficients != 0)
            eliminated = total_features - non_zero
            percentage_reduction = (eliminated / total_features) * 100
            
            st.write(f"**Alpha: {alpha}**")
            st.write(f"- Non-zero features (selected): {non_zero}")
            st.write(f"- Features eliminated: {eliminated}")
            st.write(f"- Percentage reduction: {percentage_reduction:.2f}%")
            
            results.append({
                "Alpha": alpha,
                "Selected Features": non_zero,
                "Eliminated Features": eliminated,
                "Percentage Reduction": percentage_reduction
            })
        
        st.subheader("Summary Table")
        st.table(pd.DataFrame(results))
        
else:
    st.info("Please upload a CSV file to begin.")