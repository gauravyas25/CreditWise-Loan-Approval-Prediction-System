# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **CreditWise – Loan Approval Prediction System**
#
# ## Problem Statement
# - Loan approval is a critical decision for financial institutions.
# - Manual loan approval processes are:
#     * Time-consuming
#     * Prone to human bias
#     * Inconsistent across applicants
# - Banks need a data-driven system to:
#     * Analyze applicant details
#     * Predict whether a loan should be approved or not
#     * Reduce default risk
#
# ## Main Aim of the Project
# 1. Cleans and preprocesses real-world financial data
# 2. Performs Exploratory Data Analysis (EDA) to understand patterns
# 3. Converts categorical data into machine-readable form
# 4. Analyzes feature relationships using Correlation Heatmap
# 5. Trains multiple ML models
# 6. Compares model performance
# 7. Improves results using Feature Engineering
#
#

# %%
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CreditWise – Loan Approval Prediction",
    layout="wide"
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📊 CreditWise Dashboard")

section = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Correlation Heatmap",
        "Model Training & Evaluation",
        "Feature Engineering & Comparison",
        "Loan Approval Prediction (User Input)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Domain:** Banking & Finance  
    **ML Type:** Binary Classification  
    **Models:** Logistic Regression, KNN, Naive Bayes  
    **Primary Metric:** Precision  
    """
)

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

df = load_data()

# =========================================================
# PREPROCESSING (ONCE)
# =========================================================
df_proc = df.copy()

categorical_cols = df_proc.select_dtypes(include="object").columns
numerical_cols = df_proc.select_dtypes(include="number").columns

num_imp = SimpleImputer(strategy="mean")
df_proc[numerical_cols] = num_imp.fit_transform(df_proc[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df_proc[categorical_cols] = cat_imp.fit_transform(df_proc[categorical_cols])

df_proc.drop("Applicant_ID", axis=1, inplace=True)

le = LabelEncoder()
df_proc["Education_Level"] = le.fit_transform(df_proc["Education_Level"])
df_proc["Loan_Approved"] = le.fit_transform(df_proc["Loan_Approved"])

ohe_cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df_proc[ohe_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(ohe_cols),
    index=df_proc.index
)

df_proc = pd.concat(
    [df_proc.drop(columns=ohe_cols), encoded_df],
    axis=1
)

# =========================================================
# FINAL MODEL (USED FOR PREDICTION)
# =========================================================
X_final = df_proc.drop("Loan_Approved", axis=1)
y_final = df_proc["Loan_Approved"]

scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

final_model = LogisticRegression()
final_model.fit(X_final_scaled, y_final)

# =========================================================
# PROJECT OVERVIEW
# =========================================================
if section == "Project Overview":
    st.title("💳 CreditWise – Loan Approval Prediction System")

    st.markdown("""
    ### 🔍 Problem Statement
    Loan approval is a critical and high-risk decision for financial institutions.
    Manual evaluation processes are slow, inconsistent, and prone to bias.
    Approving risky applicants leads to financial loss, while rejecting genuine
    applicants impacts customer trust and revenue.

    ### 🎯 Project Goals
    - Automate loan approval decisions using Machine Learning
    - Reduce default risk by prioritizing **precision**
    - Analyze applicant financial and demographic data
    - Build an end-to-end, production-ready ML system

    ### 🧠 What This System Does
    - Cleans and preprocesses real-world loan data
    - Performs Exploratory Data Analysis (EDA)
    - Encodes categorical variables correctly
    - Analyzes correlations between features
    - Trains and compares multiple ML models
    - Improves performance using feature engineering
    - Allows real-time prediction using user input
    """)

# =========================================================
# DATASET OVERVIEW
# =========================================================
elif section == "Dataset Overview":
    st.title("📂 Dataset Overview")
    st.dataframe(df.head())
    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

# =========================================================
# EDA
# =========================================================
elif section == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Loan Approval Distribution")
    fig, ax = plt.subplots()
    df["Loan_Approved"].value_counts().plot.pie(
        autopct="%1.1f%%",
        labels=["Approved", "Rejected"],
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("Income Distribution")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df, x="Applicant_Income", bins=20, ax=ax[0])
    sns.histplot(df, x="Coapplicant_Income", bins=20, ax=ax[1])
    st.pyplot(fig)

    st.subheader("Financial Features vs Loan Approval")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(ax=axes[0, 0], data=df, x="Loan_Approved", y="Applicant_Income")
    sns.boxplot(ax=axes[0, 1], data=df, x="Loan_Approved", y="Credit_Score")
    sns.boxplot(ax=axes[1, 0], data=df, x="Loan_Approved", y="DTI_Ratio")
    sns.boxplot(ax=axes[1, 1], data=df, x="Loan_Approved", y="Savings")
    st.pyplot(fig)

# =========================================================
# CORRELATION
# =========================================================
elif section == "Correlation Heatmap":
    st.title("🔥 Correlation Heatmap")
    corr = df_proc.corr()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    st.pyplot(fig)

    st.subheader("Correlation with Loan Approval")
    st.dataframe(corr["Loan_Approved"].sort_values(ascending=False))

# =========================================================
# MODEL TRAINING
# =========================================================
elif section == "Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    X = df_proc.drop("Loan_Approved", axis=1)
    y = df_proc["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    st.dataframe(pd.DataFrame(results))

# =========================================================
# FEATURE ENGINEERING
# =========================================================
elif section == "Feature Engineering & Comparison":
    st.title("🧠 Feature Engineering Impact")

    df_fe = df_proc.copy()
    df_fe["DTI_Ratio_sq"] = df_fe["DTI_Ratio"] ** 2
    df_fe["Credit_Score_sq"] = df_fe["Credit_Score"] ** 2

    X = df_fe.drop(["Loan_Approved", "DTI_Ratio", "Credit_Score"], axis=1)
    y = df_fe["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("Precision", round(precision_score(y_test, y_pred), 3))
    st.metric("Recall", round(recall_score(y_test, y_pred), 3))
    st.metric("F1 Score", round(f1_score(y_test, y_pred), 3))

# =========================================================
# USER INPUT PREDICTION
# =========================================================
elif section == "Loan Approval Prediction (User Input)":
    st.title("🧾 Loan Approval Prediction")

    with st.form("loan_form"):
        age = st.number_input("Age", 18, 70, 30)
        applicant_income = st.number_input("Applicant Income", 1000, 200000, 50000)
        coapp_income = st.number_input("Coapplicant Income", 0, 100000, 0)
        loan_amount = st.number_input("Loan Amount", 1000, 500000, 150000)
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        savings = st.number_input("Savings", 0, 1000000, 20000)

        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business"])
        employer_category = st.selectbox("Employer Category", ["Private", "Government"])

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame([{
            "Age": age,
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapp_income,
            "Loan_Amount": loan_amount,
            "Credit_Score": credit_score,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Education_Level": education,
            "Gender": gender,
            "Marital_Status": marital,
            "Employment_Status": employment,
            "Property_Area": property_area,
            "Loan_Purpose": loan_purpose,
            "Employer_Category": employer_category
        }])

        input_data["Education_Level"] = le.transform(input_data["Education_Level"])
        encoded_input = ohe.transform(input_data[ohe_cols])

        encoded_input_df = pd.DataFrame(
            encoded_input,
            columns=ohe.get_feature_names_out(ohe_cols)
        )

        input_data = pd.concat(
            [input_data.drop(columns=ohe_cols), encoded_input_df],
            axis=1
        )

        input_scaled = scaler_final.transform(input_data)
        prediction = final_model.predict(input_scaled)[0]
        prob = final_model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {1-prob:.2f})")
