# Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using the IBM Telco Customer Churn dataset. The project covers end-to-end EDA, preprocessing, class imbalance handling, model training, and evaluation across multiple algorithms.

## 📊 Project Overview

Customer churn — when a customer stops using a company's service — directly impacts revenue. This project builds classification models to predict which customers are likely to churn, enabling proactive retention strategies.

**Dataset:** [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7,043 customers, 33 original features
- Target: `Churn Value` (0 = stayed, 1 = churned)
- Churn rate: ~26.5% (moderately imbalanced)

## 🔍 Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked dataset structure, duplicates (0 found), and missing values
- Fixed `Total Charges` (stored as text with 11 blank values, traced to zero-tenure customers)
- Analyzed target distribution, numerical feature distributions, outliers (0 found via IQR), and correlations
- Explored churn rate across key categorical features (Contract, Internet Service, Payment Method, Senior Citizen)

### 2. Data Preprocessing
- Removed irrelevant columns (IDs, location fields) and data leakage columns (`Churn Score`, `CLTV`, `Churn Reason`, `Churn Label`)
- Standardized inconsistent categorical values (e.g., "No internet service" → "No")
- Binary encoding for Yes/No columns, one-hot encoding for multi-category columns
- Stratified 80/20 train-test split
- Feature scaling with `StandardScaler` (fit on training data only, to avoid data leakage)

### 3. Handling Class Imbalance
Two techniques were tested and compared:
- `class_weight='balanced'` — reweights the loss function to penalize minority-class errors more
- **SMOTE** (Synthetic Minority Over-sampling) — generates synthetic minority-class samples, applied only to training data

### 4. Models Trained
| Model | Technique |
|---|---|
| Dummy Classifier | Majority-class baseline |
| Logistic Regression | Untouched, class-weighted, SMOTE, hyperparameter-tuned |
| XGBoost | Plain, SMOTE |

### 5. Evaluation
Given the class imbalance, models were evaluated using **Precision, Recall, F1-score, and ROC-AUC** rather than accuracy alone, alongside confusion matrices.

## 📈 Results

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Dummy Baseline | 0.00 | 0.00 | 0.00 |
| Logistic Regression | 0.65 | 0.57 | 0.61 |
| Logistic Regression (balanced) | 0.51 | 0.78 | 0.62 |
| **Logistic Regression (SMOTE)** | **0.54** | **0.75** | **0.63** |
| XGBoost | 0.61 | 0.54 | 0.57 |
| XGBoost + SMOTE | 0.56 | 0.66 | 0.61 |

**Final Model: Logistic Regression + SMOTE** — selected for the highest F1-score among all tested models and techniques.

### Key Insight
More complex models (XGBoost) did not outperform a properly tuned, imbalance-corrected Logistic Regression — suggesting churn's relationship with key features (tenure, contract type, monthly charges) is largely linear, which a simpler model captures just as effectively.

### Top Churn Drivers
- Month-to-month contracts (vs. one/two-year contracts)
- Short customer tenure
- Fiber-optic internet service
- Higher monthly charges

## 🛠️ Tech Stack
- Python 3
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- matplotlib, seaborn

## 📁 Repository Structure
