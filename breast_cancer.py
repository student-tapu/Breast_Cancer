import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# --- Page Layout ---
st.set_page_config(page_title="Breast Cancer Detector", layout="centered")
st.title("🩺 Breast Cancer Diagnostic System")
st.write("Enter the clinical measurements below to analyze the tumor.")

# --- Load and Train Model (Internal Logic) ---
@st.cache_resource
def train_model():
    # Loading your data
    df = pd.read_csv('data.csv')
    df = df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
    
    # Encoding the diagnosis (M = 1, B = 0)
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    model = DecisionTreeClassifier(random_state=2)
    model.fit(X, y)
    return model, X.columns, le

model, feature_names, label_encoder = train_model()

# --- User Input Section ---
st.subheader("Patient Clinical Data")

# Creating a 2-column layout for input fields
col1, col2 = st.columns(2)

input_data = {}
for i, col_name in enumerate(feature_names):
    # Split inputs between column 1 and column 2
    with col1 if i % 2 == 0 else col2:
        # Use a default value from the original dataset's average to make it easier
        input_data[col_name] = st.number_input(f"{col_name.replace('_', ' ').title()}", value=0.0)

# --- Analysis Logic ---
if st.button("Analyze Results", type="primary"):
    # Convert inputs to the correct format for the model
    features = pd.DataFrame([input_data])
    
    prediction = model.predict(features)
    result = label_encoder.inverse_transform(prediction)[0]
    
    st.markdown("---")
    if result == 'M':
        st.error(f"### Result: Malignant (Cancerous)")
        st.write("The clinical measurements suggest a high probability of malignancy. Please consult a specialist immediately.")
    else:
        st.success(f"### Result: Benign (Non-Cancerous)")
        st.write("The clinical measurements suggest the tumor is benign.")
