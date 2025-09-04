# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")

# -------------------------
# Title
# -------------------------
st.markdown(
    """
    <h1 style="text-align:center; color:#2E86C1;">🩺 Diabetes Prediction App</h1>
    <p style="text-align:center; font-size:18px;">
    Powered by <b>K-Nearest Neighbors (KNN)</b> model trained on the Pima Indians Diabetes dataset.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

df = load_data()

# -------------------------
# Train Model
# -------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧑‍⚕️ Prediction", "📊 Model Performance"])

# -------------------------
# Home Page
# -------------------------
if page == "🏠 Home":
    st.subheader("📖 About this App")
    st.write("""
    This web app predicts whether a patient is likely to have diabetes 
    using **K-Nearest Neighbors (KNN)** classification.  
    - Dataset: Pima Indians Diabetes Database  
    - Features: Glucose, Blood Pressure, BMI, Age, etc.  
    - Output: **Positive (1)** or **Negative (0)** for diabetes  
    """)
    st.write("📊 **Model Accuracy:**", round(acc * 100, 2), "%")

    st.markdown("---")
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

# -------------------------
# Prediction Page
# -------------------------
elif page == "🧑‍⚕️ Prediction":
    st.subheader("🧑‍⚕️ Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
        bp = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, step=1)

    if st.button("🔍 Predict"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = knn.predict(input_scaled)

        st.markdown("---")
        if prediction[0] == 1:
            st.markdown(
                """
                <div style="padding:20px; background-color:#F1948A; border-radius:10px; text-align:center;">
                <h2>⚠️ Result: Diabetes Positive</h2>
                <p style="font-size:18px;">The model predicts this patient is <b>likely diabetic</b>.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="padding:20px; background-color:#82E0AA; border-radius:10px; text-align:center;">
                <h2>✅ Result: No Diabetes</h2>
                <p style="font-size:18px;">The model predicts this patient is <b>not diabetic</b>.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# -------------------------
# Model Performance Page
# -------------------------
elif page == "📊 Model Performance":
    st.subheader("📊 Model Performance Metrics")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
