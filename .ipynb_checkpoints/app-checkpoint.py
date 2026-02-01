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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# %%
df = pd.read_csv("loan_data.csv")
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %% [markdown]
# # Data Handling
# 1. Entire dataset contains 2 types of values
#     - Categorical Data : referred to as "objects". Eg : Gender, Property_Area, Employer_Category
#     - Numerical Data : referred to as "float64". Eg : Age, Loan_Amount, etc
# 2. Now to fill missing values in these data types, we will use statistical methods
# 3. For Categorical Data, we will use Mode : which will use the more no. of entries to fill the missing values
#     - If no. of males are more than females, Male will be  used to fill the missing values
# 3. For Numerical Data, we will use Mean : Calculate the mean of entire column/feature and fill in missing values

# %%
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["number"]).columns
numerical_cols

# %%
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df [numerical_cols])

# %%
cat_imp = SimpleImputer(strategy="most_frequent")
df [categorical_cols] = cat_imp. fit_transform(df [categorical_cols])

# %%
df.head()

# %%
df.isnull().sum()

# %% [markdown]
# # Exploratory Data Analysis

# %%
classes_count = df ["Loan_Approved"].value_counts()

plt.pie(classes_count, labels=["No", "Yes"], autopct="%1.1f%%")
plt.title("Is Loan approved or not?")

# %% [markdown]
# This pie chart shows us a distribution of percentage of loans approved and not approved
# Only around 30% of loan applications were approved

# %%
gender_cnt = df ["Gender"] .value_counts()
ax = sns.barplot(gender_cnt)
ax.bar_label(ax.containers[0])

edu_cnt = df ["Education_Level"].value_counts()
ax = sns.barplot(edu_cnt)
ax.bar_label(ax.containers[1])

# %% [markdown]
# This bar graph shows us the different categories that put up a loan applicaiton 

# %%
sns.histplot(
    data = df,
    x = "Applicant_Income",
    bins=20
)

# %%
sns.histplot(
    data = df,
    x = "Coapplicant_Income",
    bins=20
)

# %%
fig, axes = plt.subplots(2,2)

sns.boxplot(ax=axes [0, 0], data=df, x="Loan_Approved", y="Applicant_Income")
sns.boxplot(ax=axes [0, 1], data=df, x="Loan_Approved", y="Credit_Score")
sns.boxplot(ax=axes [1, 0], data=df, x="Loan_Approved", y="DTI_Ratio")
sns.boxplot(ax=axes [1, 1], data=df, x="Loan_Approved", y="Savings")



plt.tight_layout()

# %%
sns.histplot(
    data=df,
    x="Credit_Score",
    hue="Loan_Approved",
    bins=20,
    multiple="dodge"
)

# %%
df = df.drop("Applicant_ID", axis=1)

# %% [markdown]
# # Encoding Data
# 1. Label Encoding : Converts categorical data into numerical data within the data column itself
#     - Consider Example, Gender column has M & F values, it will assign M = 0, F = 1 or vice-versa
#     - This encoding is used only when we are working on Ordinal Data which tells us an order or ranking via numbers. If Male = 1 and Female = 0, then male is superior than female 
#
#   
# 2. One Hot Encoding : Creates two seperate columns for Male and Female values like Gender_Male, Gender_Female
#     - Used only on Nominal Data which has no order or ranking in data

# %%
le = LabelEncoder()
df ["Education_Level"] = le. fit_transform(df["Education_Level"])
df ["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])

# %%
cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(encoded, columns = ohe.get_feature_names_out(cols), index = df.index)

# %%
df = pd.concat([df.drop(columns = cols), encoded_df], axis=1)

# %%
df.columns

# %% [markdown]
# # Correlation Heatmap

# %%
num_cols = df.select_dtypes(include="number")
corr_matrix = num_cols. corr()

num_cols.corr()["Loan_Approved"].sort_values(ascending=False)

# %%
plt.figure(figsize=(15, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

# %% [markdown]
# # Training & Testing

# %%
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# %%
X.head()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
X_train.head()

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # Logistic Regression

# %%
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# Evaluation
print("Logistic Regression Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))

print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))

# %% [markdown]
# # k - Nearest Neighbors

# %%
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

# Evaluation
print("KNN Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))

# %% [markdown]
# # Naive Bayes Model

# %%
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

# Evaluation
print("Naive Bayes Model")
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("F1 score: ", f1_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("CM: ", confusion_matrix(y_test, y_pred))

# %% [markdown]
# # Fearure Engineering to Improve Models Performance

# %%
df["DTI_Ratio_sq"] = df ["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2


X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df ["Loan_Approved"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

# Evaluation
print("Logistic Regression Model")
lr_precision = precision_score(y_test, y_pred)
lr_recall = recall_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)
lr_accuracy = accuracy_score(y_test, y_pred)
print("CM: ", confusion_matrix(y_test, y_pred))

# %%
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

# Evaluation
print("KNN Model")
knn_precision = precision_score(y_test, y_pred)
knn_recall = recall_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred)
knn_accuracy = accuracy_score(y_test, y_pred)
print("CM: ", confusion_matrix(y_test, y_pred))

# %%
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

# Evaluation
print("Naive Bayes Model")
nb_precision = precision_score(y_test, y_pred)
nb_recall = recall_score(y_test, y_pred)
nb_f1 = f1_score(y_test, y_pred)
nb_accuracy = accuracy_score(y_test, y_pred)
print("CM: ", confusion_matrix(y_test, y_pred))

# %%
performance_df = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN", "Naive Bayes"],
    "Accuracy": [lr_accuracy, knn_accuracy, nb_accuracy],
    "Precision": [lr_precision, knn_precision, nb_precision],
    "Recall": [lr_recall, knn_recall, nb_recall],
    "F1 Score": [lr_f1, knn_f1, nb_f1]
})

performance_df

# %%
plt.figure()
plt.bar(performance_df["Model"], performance_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.show()

# %%
plt.figure()
plt.bar(performance_df["Model"], performance_df["Precision"])
plt.title("Model Precision Comparison")
plt.xlabel("Model")
plt.ylabel("Precision")
plt.show()

# %%
plt.figure()
plt.bar(performance_df["Model"], performance_df["Recall"])
plt.title("Model Recall Comparison")
plt.xlabel("Model")
plt.ylabel("Recall")
plt.show()


# %%
plt.figure()
plt.bar(performance_df["Model"], performance_df["F1 Score"])
plt.title("Model F1-Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.show()

# %%

# %%

# %%
