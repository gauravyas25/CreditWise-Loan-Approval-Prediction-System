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

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

df = load_data()

# =========================================================
# PREPROCESSING
# =========================================================
df_proc = df.copy()

categorical_cols = df_proc.select_dtypes(include="object").columns
numerical_cols = df_proc.select_dtypes(include="number").columns

num_imp = SimpleImputer(strategy="mean")
df_proc[numerical_cols] = num_imp.fit_transform(df_proc[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df_proc[categorical_cols] = cat_imp.fit_transform(df_proc[categorical_cols])

df_proc.drop("Applicant_ID", axis=1, inplace=True)

edu_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df_proc["Education_Level"] = edu_encoder.fit_transform(df_proc["Education_Level"])
df_proc["Loan_Approved"] = target_encoder.fit_transform(df_proc["Loan_Approved"])

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
# FINAL MODEL
# =========================================================
X_final = df_proc.drop("Loan_Approved", axis=1)
y_final = df_proc["Loan_Approved"]

feature_order = X_final.columns.tolist()

scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

final_model = LogisticRegression()
final_model.fit(X_final_scaled, y_final)

coef_df = pd.DataFrame({
    "Feature": feature_order,
    "Coefficient": final_model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

# =========================================================
# PROJECT OVERVIEW
# =========================================================
if section == "Project Overview":
    st.title("💳 CreditWise – Loan Approval Prediction System")

    st.markdown("""
    ### 🔍 Problem Statement
    Loan approval is a high-stakes decision in banking where incorrect approvals
    lead to financial losses and incorrect rejections lead to loss of genuine customers.
    Manual evaluation processes are slow, inconsistent, and prone to bias.

    ### 🎯 Project Goals
    - Automate loan approval using Machine Learning
    - Reduce default risk by prioritizing precision
    - Analyze financial and demographic attributes of applicants
    - Build an interpretable and end-to-end ML system

    ### 🧠 End-to-End Pipeline
    Data Collection → Data Cleaning → Exploratory Data Analysis →  
    Encoding & Scaling → Model Training → Feature Engineering →  
    Model Comparison → Prediction with Explanation
    """)

# =========================================================
# DATASET OVERVIEW
# =========================================================
elif section == "Dataset Overview":
    st.title("📂 Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Missing Values Analysis")
    st.markdown("""
    This table shows the count of missing values in each column.
    Numerical features are later filled using **mean imputation**,
    while categorical features are filled using **mode imputation**.
    """)
    st.dataframe(df.isnull().sum())

# =========================================================
# EDA
# =========================================================
elif section == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.markdown("""
    Exploratory Data Analysis helps in understanding:
    - Class imbalance
    - Distribution of income and financial attributes
    - Differences between approved and rejected loans
    """)

    st.subheader("Loan Approval Distribution (Pie Chart)")
    st.markdown("""
    This pie chart shows the proportion of approved vs rejected loans.
    It highlights class imbalance, which is important because imbalanced data
    can bias models towards the majority class.
    """)

    fig, ax = plt.subplots()
    df["Loan_Approved"].value_counts().plot.pie(
        autopct="%1.1f%%",
        labels=["Approved", "Rejected"],
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    st.subheader("Income Distribution (Histograms)")
    st.markdown("""
    These histograms show how applicant and co-applicant incomes are distributed.
    They help identify skewness, income concentration, and outliers.
    """)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df, x="Applicant_Income", bins=20, ax=ax[0])
    sns.histplot(df, x="Coapplicant_Income", bins=20, ax=ax[1])
    st.pyplot(fig)

    st.subheader("Financial Features vs Loan Approval (Box Plots)")
    st.markdown("""
    Box plots compare financial attributes across approved and rejected loans.
    They help identify which features show clear separation between classes.
    """)

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

    st.markdown("""
    The correlation heatmap shows the linear relationship between numerical features.
    Values closer to +1 indicate strong positive correlation, while values near -1
    indicate strong negative correlation.

    This helps in:
    - Identifying important predictors
    - Detecting multicollinearity
    - Understanding feature influence on loan approval
    """)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(df_proc.corr(), cmap="coolwarm", annot=False)
    st.pyplot(fig)

    st.subheader("Correlation with Loan Approval")
    st.markdown("""
    This table ranks features based on their correlation with the target variable.
    Features with higher absolute correlation values tend to have more influence
    on the model's decision.
    """)
    st.dataframe(df_proc.corr()["Loan_Approved"].sort_values(ascending=False))

# =========================================================
# MODEL TRAINING
# =========================================================
elif section == "Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    st.markdown("""
    Multiple classification models are trained and evaluated.
    Precision is emphasized because false positives (approving risky applicants)
    are more costly in loan approval systems.
    """)

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

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred)
        })

    st.dataframe(pd.DataFrame(rows))

# =========================================================
# FEATURE ENGINEERING
# =========================================================
elif section == "Feature Engineering & Comparison":
    st.title("🧠 Feature Engineering Impact")

    st.markdown("""
    Feature engineering introduces non-linear representations of existing features.
    Squared terms help the model capture non-linear relationships between
    financial risk and loan approval.
    """)

    df_fe = df_proc.copy()
    df_fe["DTI_Ratio_sq"] = df_fe["DTI_Ratio"] ** 2
    df_fe["Credit_Score_sq"] = df_fe["Credit_Score"] ** 2

    st.markdown("""
    **What changed after feature engineering:**
    - Higher sensitivity to extreme DTI values
    - Stronger emphasis on very high or very low credit scores
    - Improved separation between risky and safe applicants
    """)

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
# USER INPUT + EXPLANATION
# =========================================================
elif section == "Loan Approval Prediction (User Input)":
    st.title("🧾 Loan Approval Prediction with Explanation")

    with st.form("loan_form"):
        age = st.number_input("Age", 18, 70, 30)
        applicant_income = st.number_input("Applicant Income", 1000, 200000, 50000)
        coapp_income = st.number_input("Coapplicant Income", 0, 100000, 0)
        loan_amount = st.number_input("Loan Amount", 1000, 500000, 150000)
        credit_score = st.number_input("Credit Score", 300, 900, 700)
        dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)
        savings = st.number_input("Savings", 0, 1000000, 20000)

        education = st.selectbox("Education Level", edu_encoder.classes_)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married"])
        employment = st.selectbox("Employment Status", ["Salaried", "Self-Employed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Business"])
        employer_category = st.selectbox("Employer Category", ["Private", "Government"])

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([{
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

        input_df["Education_Level"] = edu_encoder.transform(input_df["Education_Level"])
        encoded_input = ohe.transform(input_df[ohe_cols])
        encoded_input_df = pd.DataFrame(
            encoded_input,
            columns=ohe.get_feature_names_out(ohe_cols)
        )

        input_df = pd.concat(
            [input_df.drop(columns=ohe_cols), encoded_input_df],
            axis=1
        )

        input_df = input_df.reindex(columns=feature_order, fill_value=0)

        input_scaled = scaler_final.transform(input_df)
        pred = final_model.predict(input_scaled)[0]
        prob = final_model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.success(f"✅ Loan Approved (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Loan Rejected (Confidence: {1 - prob:.2f})")

        st.subheader("📌 Decision Explanation")
        contrib = coef_df.copy()
        contrib["Contribution"] = contrib["Coefficient"] * input_scaled[0]
        contrib = contrib.sort_values(
            by="Contribution",
            ascending=(pred == 0)
        ).head(6)

        st.dataframe(contrib[["Feature", "Contribution"]])
