# 💳 CreditWise – Loan Approval Prediction System

🔗 **Live Application**  
https://creditwise-loan-approval-prediction-system-1433.streamlit.app/

---

## 📌 Project Overview
- End-to-end Machine Learning system for loan approval prediction
- Automates decision-making for financial institutions
- Focuses on reducing risky loan approvals
- Deployed as an interactive Streamlit application

---

## ❓ Problem Statement
- Manual loan approval is slow and inconsistent
- High dependency on human judgment introduces bias
- Incorrect approvals lead to financial loss
- Need for a data-driven, explainable decision system

---

## 🎯 Project Objectives
- Clean and preprocess real-world financial data
- Perform Exploratory Data Analysis (EDA)
- Encode categorical features correctly
- Analyze feature relationships using correlation
- Train and compare multiple ML models
- Improve performance using feature engineering
- Prioritize Precision to reduce default risk

---

## 📂 Dataset Description
### Numerical Features
- Age
- Applicant Income
- Coapplicant Income
- Loan Amount
- Credit Score
- DTI Ratio
- Savings

### Categorical Features
- Gender
- Marital Status
- Education Level
- Employment Status
- Employer Category
- Loan Purpose
- Property Area

### Target Variable
- Loan Approved (Yes / No)

---

## 🧹 Data Preprocessing
- Mean imputation for numerical features
- Mode imputation for categorical features
- Removed non-predictive Applicant ID
- Label Encoding for ordinal features
- One-Hot Encoding for nominal features
- Feature scaling using StandardScaler

---

## 📊 Exploratory Data Analysis (EDA)
- Pie chart to analyze loan approval distribution
- Histograms to study income distributions
- Box plots to compare financial features against loan approval
- Identification of outliers and class imbalance

---

## 🔥 Correlation Analysis
- Correlation heatmap for numerical features
- Identification of strong positive and negative relationships
- Ranking of features based on correlation with loan approval
- Detection of multicollinearity

---

## 🤖 Machine Learning Models Used
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Naive Bayes

---

## 📏 Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score

### Metric Focus
- Precision prioritized to avoid risky loan approvals

---

## 🧠 Feature Engineering
- Squared DTI Ratio feature
- Squared Credit Score feature
- Captured non-linear financial risk patterns
- Improved Precision and F1-Score

---

## 🛠️ Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

## 🌟 Key Highlights
- Real-world finance domain project
- End-to-end ML pipeline
- Business-oriented evaluation strategy
- Explainable model decisions
- Deployed and production-ready

---

## 👨‍💻 Author
**Gaurav Vyas**  
Machine Learning & Full-Stack Developer
