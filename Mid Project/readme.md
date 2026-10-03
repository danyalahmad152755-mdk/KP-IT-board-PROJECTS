# Telco Customer Churn Prediction

A machine learning project to predict customer churn for a telecom company using the IBM Telco Customer Churn dataset. The project covers end-to-end EDA, preprocessing, class imbalance handling, model training, and evaluation.

## 📊 Project Overview

Customer churn — when a customer stops using a company's service — directly impacts revenue. This project builds classification models to predict which customers are likely to churn, enabling proactive retention strategies.

**Dataset:** IBM Telco Customer Churn

* 7,043 customers, 33 original features
* Target: `Churn Value` (0 = stayed, 1 = churned)
* Churn rate: ~26.5% (moderately imbalanced)

## 🔍 Workflow

### 1. Exploratory Data Analysis (EDA)

* Checked dataset structure, duplicates (0 found), and missing values
* Fixed `Total Charges` (stored as text with 11 blank values, traced to zero-tenure customers, filled with 0)
* Analyzed target distribution, numerical feature distributions, outliers (0 found via IQR method), and correlations
* Explored churn rate across key categorical features (Contract, Internet Service, Payment Method, Senior Citizen)

### 2. Data Preprocessing

* Removed irrelevant columns (IDs, location fields) and data leakage columns (`Churn Score`, `CLTV`, `Churn Reason`, `Churn Label`)
* Standardized inconsistent categorical values (e.g., `"No internet service"` → `"No"`)
* Binary encoding for Yes/No columns, one-hot encoding for multi-category columns
* Stratified 80/20 train-test split
* Feature scaling with `StandardScaler` (fit on training data only, to avoid data leakage)

### 3. Handling Class Imbalance

Two techniques were tested and compared:

* `class_weight='balanced'` — reweights the loss function to penalize minority-class (churn) errors more heavily
* **SMOTE** (Synthetic Minority Over-sampling) — generates synthetic minority-class samples, applied only to training data

Hyperparameter tuning was performed on the SMOTE-trained Logistic Regression using `GridSearchCV`, with SMOTE correctly wrapped inside an `imblearn` Pipeline to prevent data leakage across cross-validation folds.

### 4. Models Trained

| Model               | Technique               |
| ------------------- | ----------------------- |
| Dummy Classifier    | Majority-class baseline |
| Logistic Regression | Class-weighted, SMOTE   |
| XGBoost             | Plain, SMOTE            |

### 5. Evaluation

Given the class imbalance, models were evaluated using **Accuracy, Precision, Recall, F1-score, and ROC-AUC** rather than accuracy alone, alongside confusion matrices.

## 📈 Results

| Model                           |  Accuracy | Precision | Recall |        F1 | ROC-AUC |
| ------------------------------- | --------: | --------: | -----: | --------: | ------: |
| Dummy Baseline                  |     0.735 |     0.000 |  0.000 |     0.000 |   0.500 |
| Logistic Regression (balanced)  |     0.745 |     0.513 |  0.778 |     0.618 |   0.849 |
| **Logistic Regression (SMOTE)** | **0.761** |     0.535 |  0.751 | **0.625** |   0.840 |
| XGBoost                         |     0.786 |     0.608 |  0.543 |     0.573 |   0.834 |
| XGBoost + SMOTE                 |     0.769 |     0.553 |  0.668 |     0.605 |   0.824 |

**Final Model: Logistic Regression + SMOTE**

The Logistic Regression + SMOTE model was selected based on the highest F1-score (0.625) among the tested models and techniques, providing a balance between precision and recall for the churn class.

### Key Insight

A properly tuned, imbalance-corrected Logistic Regression achieved a higher F1-score than the tested XGBoost configurations, despite XGBoost's higher raw accuracy. This suggests that a substantial portion of the churn signal in the tested feature set can be captured by a relatively simple and interpretable linear model.

### Baseline Validation

A Dummy Classifier (always predicting "No Churn") achieved 73.5% accuracy but 0.000 across precision, recall, and F1 for churn. This demonstrates why accuracy alone can be misleading when evaluating an imbalanced classification problem.

### Top Churn Drivers

* Month-to-month contracts (vs. one/two-year contracts)
* Short customer tenure
* Fiber-optic internet service
* Higher monthly charges

## 🛠️ Tech Stack

* Python 3
* pandas
* numpy
* scikit-learn
* imbalanced-learn (SMOTE)
* XGBoost
* matplotlib
* seaborn

## 📁 Repository Structure

```text
Telco-Customer-Churn/
│
├── churn_prediction.ipynb
├── requirements.txt
├── README.md
└── dataset/
    └── telco_customer_churn.csv
```

## ⚙️ How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch the Jupyter Notebook:

```bash
jupyter notebook churn_prediction.ipynb
```

## 📋 Requirements

The project requires the following Python libraries:

```text
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
matplotlib
seaborn
jupyter
```

## ⚠️ Limitations

* XGBoost was evaluated with lightly tuned hyperparameters; a more exhaustive search could change its performance.
* The classification threshold was kept at the default 0.5; business-specific threshold tuning could further optimize the precision-recall trade-off.
* The project uses a single train-test split for final evaluation.

## 👤 Author

**Danyal Ahmad**

Machine Learning / Data Science Student
