import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="CreditWise Loan Approval System",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🏦 CreditWise – Loan Approval Prediction System")

# -----------------------------------
# PROJECT INFORMATION
# -----------------------------------
with st.expander("📌 Project Overview", expanded=True):
    st.markdown("""
    **Project Name:** CreditWise – Loan Approval System  

    **Problem Statement:**  
    Financial institutions receive thousands of loan applications daily.  
    Manual verification is time-consuming, error-prone, and inconsistent.  
    This project aims to automate loan approval decisions using Machine Learning.

    **Aim:**  
    To predict whether a loan application should be **Approved (Y)** or **Rejected (N)** based on applicant details.

    **Target Variable:**  
    `Loan_Status`

    **Tech Stack:**  
    - Python  
    - Pandas, NumPy  
    - Matplotlib, Seaborn  
    - Scikit-learn  
    - Streamlit
    """)

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

df = load_data()

# -----------------------------------
# DATA PREVIEW
# -----------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# EDA SECTION
# -----------------------------------
st.subheader("📊 Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Loan Status Distribution**")
    fig, ax = plt.subplots()
    df["Loan_Status"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Loan Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.markdown("**Education Level Distribution**")
    fig, ax = plt.subplots()
    df["Education"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Education")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Histogram
st.markdown("**Applicant Income Distribution**")
fig, ax = plt.subplots()
sns.histplot(df["ApplicantIncome"], kde=True, ax=ax)
st.pyplot(fig)

# Boxplot
st.markdown("**Applicant Income vs Loan Status**")
fig, ax = plt.subplots()
sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=df, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.markdown("**Correlation Heatmap**")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------------
# DATA PREPROCESSING
# -----------------------------------
data = df.copy()

label_encoders = {}
for col in data.select_dtypes(include="object").columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------
# MODELS
# -----------------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

# -----------------------------------
# MODEL COMPARISON
# -----------------------------------
st.subheader("⚖️ Model Performance Comparison")

results_df = pd.DataFrame(results).T
st.dataframe(results_df)

model_choice = st.selectbox(
    "Select Model for Prediction",
    list(models.keys())
)

selected_model = models[model_choice]

# -----------------------------------
# SIDEBAR – USER INPUT
# -----------------------------------
st.sidebar.header("🧾 Enter Applicant Details")

user_input = {}
for col in df.drop("Loan_Status", axis=1).columns:
    if df[col].dtype == "object":
        user_input[col] = st.sidebar.selectbox(col, df[col].unique())
    else:
        user_input[col] = st.sidebar.number_input(col, float(df[col].min()), float(df[col].max()))

input_df = pd.DataFrame([user_input])

for col in input_df.select_dtypes(include="object").columns:
    input_df[col] = label_encoders[col].transform(input_df[col])

input_scaled = scaler.transform(input_df)

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.sidebar.button("🔍 Predict Loan Status"):
    prediction = selected_model.predict(input_scaled)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.sidebar.success(f"Loan Status: {result}")

# -----------------------------------
# EXPLANATION TAB
# -----------------------------------
st.subheader("🧠 Explanation (Interview-Ready)")

with st.expander("Why Precision is Important?"):
    st.markdown("""
    Precision is crucial in loan approval systems because:
    - False positives (approving risky loans) cause **financial loss**
    - Banks prefer **safe approvals over mass approvals**
    """)

with st.expander("Why Multiple Models?"):
    st.markdown("""
    - Logistic Regression → Baseline, interpretable
    - Decision Tree → Rule-based decisions
    - Random Forest → Handles non-linearity & reduces overfitting
    - Naive Bayes → Fast, probabilistic
    """)

with st.expander("Real-World Use Case"):
    st.markdown("""
    This system can be integrated into:
    - Bank loan portals
    - NBFC risk engines
    - FinTech credit scoring pipelines
    """)
