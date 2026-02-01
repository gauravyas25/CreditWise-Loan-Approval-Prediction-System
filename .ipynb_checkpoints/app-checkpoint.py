import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CreditWise – Loan Approval Prediction",
    layout="wide"
)

st.title("🏦 CreditWise – Loan Approval Prediction System")

# =========================================================
# FIXED SCHEMA (DO NOT CHANGE)
# =========================================================
FEATURE_COLUMNS = [
    'Applicant_Income',
    'Coapplicant_Income',
    'Age',
    'Dependents',
    'Credit_Score',
    'Existing_Loans',
    'DTI_Ratio',
    'Savings',
    'Collateral_Value',
    'Loan_Amount',
    'Loan_Term',
    'Education_Level',
]

TARGET_COLUMN = "Loan_Approved"
EXPECTED_COLUMNS = set(FEATURE_COLUMNS + [TARGET_COLUMN])

# =========================================================
# LOAD DATA (STREAMLIT CLOUD SAFE)
# =========================================================
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "loan_data.csv")
    return pd.read_csv(path)

df = load_data()

# =========================================================
# STRICT SCHEMA VALIDATION
# =========================================================
missing_cols = EXPECTED_COLUMNS - set(df.columns)
extra_cols = set(df.columns) - EXPECTED_COLUMNS

if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

if extra_cols:
    st.warning(f"⚠️ Extra columns detected (ignored): {extra_cols}")

# =========================================================
# PROJECT OVERVIEW
# =========================================================
with st.expander("📌 Project Overview", expanded=True):
    st.markdown("""
**Project Name:** CreditWise – Loan Approval Prediction System  

**Problem Statement:**  
Manual loan approval is slow and prone to risk. Financial institutions need
data-driven systems to minimize defaults and ensure consistency.

**Aim:**  
To predict whether a loan application should be **Approved (1)** or
**Rejected (0)** using financial, demographic, and behavioral features.

**Target Variable:** `Loan_Approved`

**Key Focus:**  
- Risk reduction  
- Precision-oriented decision making  
- Production-safe deployment
""")

# =========================================================
# DATA PREVIEW
# =========================================================
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# =========================================================
# EDA
# =========================================================
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df[TARGET_COLUMN].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Loan Approval Distribution")
    ax.set_xlabel("Loan Approved")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.histplot(df["Credit_Score"], kde=True, ax=ax)
    ax.set_title("Credit Score Distribution")
    st.pyplot(fig)

fig, ax = plt.subplots()
sns.boxplot(x=TARGET_COLUMN, y="DTI_Ratio", data=df, ax=ax)
ax.set_title("DTI Ratio vs Loan Approval")
st.pyplot(fig)

numeric_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].select_dtypes(
    include=[np.number]
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    ax=ax
)
ax.set_title("Correlation Heatmap (Numeric Features Only)")
st.pyplot(fig)

# =========================================================
# TRAIN–TEST SPLIT
# =========================================================
X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# MODELS
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds)
    }

# =========================================================
# MODEL COMPARISON
# =========================================================
st.subheader("⚖️ Model Performance Comparison")
st.dataframe(pd.DataFrame(metrics).T)

selected_model_name = st.selectbox(
    "Select Model for Prediction",
    list(models.keys())
)
selected_model = models[selected_model_name]

# =========================================================
# SIDEBAR INPUT (SCHEMA-AWARE, NO RENAMING)
# =========================================================
st.sidebar.header("🧾 Applicant Details")

input_data = {col: 0 for col in FEATURE_COLUMNS}

for col in FEATURE_COLUMNS:
    if col.endswith("_sq"):
        continue

    if df[col].nunique() <= 2:
        input_data[col] = st.sidebar.selectbox(col, [0, 1])
    else:
        input_data[col] = st.sidebar.number_input(col, value=float(df[col].median()))

# Derived features
input_data["DTI_Ratio_sq"] = input_data["DTI_Ratio"] ** 2
input_data["Credit_Score_sq"] = input_data["Credit_Score"] ** 2

# =========================================================
# PREDICTION
# =========================================================
if st.sidebar.button("🔍 Predict Loan Approval"):
    input_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    input_scaled = scaler.transform(input_df)
    prediction = selected_model.predict(input_scaled)[0]

    if prediction == 1:
        st.sidebar.success("✅ Loan Approved")
    else:
        st.sidebar.error("❌ Loan Rejected")

# =========================================================
# EXPLANATION (INTERVIEW-READY)
# =========================================================
st.subheader("🧠 Explanation")

with st.expander("Why Precision is Critical"):
    st.markdown("""
In loan approval systems, false positives are expensive.
Approving a risky applicant leads to financial loss, so
precision is prioritized over recall.
""")

with st.expander("Why Fixed Schema is Mandatory"):
    st.markdown("""
The model was trained on a fixed, feature-engineered dataset.
Enforcing the same schema during inference prevents
training–serving skew and ensures reliable predictions.
""")

with st.expander("Real-World Applicability"):
    st.markdown("""
This system can be deployed in:
- Bank loan processing pipelines  
- FinTech credit scoring engines  
- Automated risk assessment systems
""")
