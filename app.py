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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="CreditWise – Loan Approval Prediction",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📊 CreditWise Navigation")

section = st.sidebar.radio(
    "Go to Section:",
    [
        "Project Overview",
        "Dataset Overview",
        "Exploratory Data Analysis",
        "Correlation Analysis",
        "Model Training & Evaluation",
        "Feature Engineering & Comparison"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Developed By:** Gaurav Vyas  
    **Tech Stack:** Python, Pandas, NumPy, Seaborn, Scikit-Learn, Streamlit  
    **Domain:** Finance / Banking / Risk Analytics
    """
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

df = load_data()

# --------------------------------------------------
# Project Overview
# --------------------------------------------------
if section == "Project Overview":
    st.title("💳 CreditWise – Loan Approval Prediction System")

    st.markdown("""
    ### 🔍 Problem Statement
    Loan approval is a high-risk, high-impact decision in the banking domain.
    Traditional manual evaluation is:
    - Time-consuming  
    - Prone to bias  
    - Inconsistent across applicants  

    **CreditWise** introduces a **data-driven ML approach** to predict loan approval
    using applicant demographics, financial metrics, and credit history.
    """)

    st.markdown("""
    ### 🎯 Objectives
    - Clean and preprocess real-world financial data
    - Perform deep Exploratory Data Analysis (EDA)
    - Encode categorical features correctly
    - Analyze feature relationships using correlation
    - Train and compare multiple ML models
    - Improve performance via feature engineering
    """)

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
elif section == "Dataset Overview":
    st.title("📂 Dataset Overview")

    st.subheader("Raw Dataset")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(df.info())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum())

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
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

ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False)

encoded = ohe.fit_transform(df_proc[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))

df_proc = pd.concat([df_proc.drop(columns=ohe_cols), encoded_df], axis=1)

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif section == "Exploratory Data Analysis":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Loan Approval Distribution")
    fig, ax = plt.subplots()
    df["Loan_Approved"].value_counts().plot.pie(
        autopct="%1.1f%%",
        labels=["Approved", "Rejected"],
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("Applicant Income Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df, x="Applicant_Income", bins=20, ax=ax)
    st.pyplot(fig)

    st.subheader("Credit Score vs Loan Approval")
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Loan_Approved", y="Credit_Score", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# Correlation Analysis
# --------------------------------------------------
elif section == "Correlation Analysis":
    st.title("🔥 Correlation Heatmap")

    corr = df_proc.corr()

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)

    st.subheader("Top Correlated Features with Loan Approval")
    st.dataframe(
        corr["Loan_Approved"].sort_values(ascending=False)
    )

# --------------------------------------------------
# Model Training
# --------------------------------------------------
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

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
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

    st.metric("Improved F1 Score", round(f1_score(y_test, y_pred), 3))
    st.metric("Improved Precision", round(precision_score(y_test, y_pred), 3))

# %%

# %%

# %%
