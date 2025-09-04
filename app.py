# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# -------------------------
# Title
# -------------------------
st.title("🩺 Diabetes Prediction with KNN")
st.markdown("This app uses a **K-Nearest Neighbors (KNN)** model trained on the **Pima Indians Diabetes dataset** to predict whether a patient is likely to have diabetes.")

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
# Sidebar - Model Info
# -------------------------
st.sidebar.header("⚙️ Model Information")
st.sidebar.write(f"**Accuracy on Test Set:** {acc:.2f}")
st.sidebar.write("**Algorithm:** K-Nearest Neighbors")
st.sidebar.write("**Dataset:** Pima Indians Diabetes")

# -------------------------
# Two-column Layout
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔢 Enter Patient Details")

    preg = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
    skin = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, step=1)

    if st.button("🔍 Predict"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = knn.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ The model predicts: **Diabetes Positive**")
        else:
            st.success("✅ The model predicts: **No Diabetes**")

with col2:
    st.subheader("📊 Model Performance")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
