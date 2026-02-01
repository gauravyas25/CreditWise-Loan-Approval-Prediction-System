# ============================================================
# CreditWise: Loan Approval Prediction System
# Author: Gaurav Vyas
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------- PAGE CONFIG -------------------------
st.set_page_config(
    page_title="CreditWise | Loan Approval Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- TITLE -------------------------------
st.title("💳 CreditWise: Loan Approval Prediction System")

st.markdown("""
**CreditWise** is a machine learning–based decision support system designed to
predict whether a loan application will be **approved or rejected** based on
applicant financial and demographic attributes.
""")

# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander("📘 Project Overview", expanded=True):
    st.markdown("""
### 🔍 Problem Statement
Financial institutions face significant risk when approving loans.
Manual evaluation is time-consuming, error-prone, and inconsistent.
There is a need for an **automated, data-driven loan approval system**
that ensures **accuracy, fairness, and risk minimization**.

### 🎯 Aim
To build a **machine learning classification system** that predicts
loan approval status using historical applicant data and evaluates
model performance using industry-relevant metrics.

### 🧠 Models Used
- **Naive Bayes (GaussianNB)**
- **Logistic Regression**

### 📊 Performance Evaluation Metrics
- Accuracy
- Precision (most critical)
- Recall
- F1-Score
- Confusion Matrix

Precision is prioritized because **false approvals (bad loans)** are
costlier than false rejections.
""")

# ============================================================
# DATA LOADING
# ============================================================

st.header("📂 Dataset Loading & Preview")

uploaded_file = st.file_uploader(
    "Upload Loan Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ========================================================
    # DATA PREPROCESSING
    # ========================================================

    st.header("🧹 Data Preprocessing")

    df_processed = df.copy()

    label_encoders = {}
    for col in df_processed.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    st.write("✔ Categorical features encoded using Label Encoding")

    # Target & Features
    X = df_processed.drop("Loan_Status", axis=1)
    y = df_processed["Loan_Status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    st.write("✔ Train-Test Split (80%-20%)")
    st.write("✔ Feature Scaling applied")

    # ========================================================
    # EXPLORATORY DATA ANALYSIS
    # ========================================================

    st.header("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Status Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=y, ax=ax)
        ax.set_title("Loan Approval Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df_processed.corr(),
            cmap="coolwarm",
            annot=False,
            ax=ax
        )
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

    st.markdown("""
### 📌 Heatmap Insight
- Strong correlation between **credit history and loan approval**
- Weak correlation for demographic features
- Confirms financial attributes dominate approval decisions
""")

    # ========================================================
    # MODEL TRAINING
    # ========================================================

    st.header("🤖 Model Training & Evaluation")

    # ----------------- Naive Bayes ---------------------------
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)

    # ---------------- Logistic Regression -------------------
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # ========================================================
    # PERFORMANCE COMPARISON
    # ========================================================

    st.subheader("📈 Model Performance Comparison")

    def evaluate_model(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred)
        }

    results = pd.DataFrame.from_dict(
        {
            "Naive Bayes": evaluate_model(y_test, y_pred_nb),
            "Logistic Regression": evaluate_model(y_test, y_pred_lr)
        },
        orient="index"
    )

    st.dataframe(results.style.highlight_max(axis=0))

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader("🧩 Confusion Matrix")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(y_test, y_pred_nb),
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_title("Naive Bayes Confusion Matrix")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(y_test, y_pred_lr),
            annot=True,
            fmt="d",
            cmap="Greens",
            ax=ax
        )
        ax.set_title("Logistic Regression Confusion Matrix")
        st.pyplot(fig)

    st.markdown("""
### 🧠 Key Insight
- **Naive Bayes** shows higher **precision**, making it safer for loan approval
- Logistic Regression performs competitively but allows more false positives
""")

else:
    st.warning("Please upload a CSV dataset to proceed.")


# ============================================================
# SIDEBAR: LOAN PREDICTION SYSTEM
# ============================================================

st.sidebar.title("🏦 Loan Approval Predictor")
st.sidebar.markdown("""
Fill in applicant details to predict **Loan Approval Status**  
(Model used: **Naive Bayes – Precision Optimized**)
""")

# ----------- Sidebar Inputs ----------------

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.sidebar.number_input(
    "Applicant Income", min_value=0, step=500
)

coapplicant_income = st.sidebar.number_input(
    "Co-applicant Income", min_value=0, step=500
)

loan_amount = st.sidebar.number_input(
    "Loan Amount (in thousands)", min_value=0, step=10
)

loan_amount_term = st.sidebar.selectbox(
    "Loan Term (months)", [360, 180, 240, 120, 60]
)

credit_history = st.sidebar.selectbox(
    "Credit History", [1.0, 0.0],
    help="1.0 = Good Credit | 0.0 = Poor Credit"
)

property_area = st.sidebar.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

# ----------- Prediction Button ----------------

if st.sidebar.button("🔍 Predict Loan Status"):

    input_data = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_employed],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_amount_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area]
    })

    # Apply Label Encoding
    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = nb_model.predict(input_scaled)[0]
    probability = nb_model.predict_proba(input_scaled)[0][prediction]

    # ----------- Output ----------------
    st.sidebar.markdown("---")

    if prediction == 1:
        st.sidebar.success(
            f"✅ **Loan Approved**\n\nConfidence: **{probability:.2%}**"
        )
    else:
        st.sidebar.error(
            f"❌ **Loan Rejected**\n\nConfidence: **{probability:.2%}**"
        )

    st.sidebar.markdown("""
    ### 📌 Decision Logic
    - Model prioritizes **Precision**
    - Reduces risk of **bad loan approvals**
    - Credit History has highest influence
    """)
