# 📊 Customer Churn Prediction: An End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/SMOTE-Enabled-brightgreen.svg)](https://imbalanced-learn.org/)

> A production-grade machine learning project focused on identifying customer churn using the Telco Customer Churn dataset.

## 📖 Overview

Customer churn is a critical metric for subscription-based businesses. Identifying high-risk customers allows companies to proactively implement retention strategies. This project implements a full ML pipeline, handling severe class imbalance via **SMOTE**, strictly maintaining data integrity through rigorous cross-validation, and yielding actionable business insights through robust feature importance analysis.

### Key Highlights
- **Proper Data Leakage Prevention:** `SMOTE` is embedded directly into the cross-validation pipeline (`imblearn.pipeline`), avoiding the common pitfall of oversampling before splitting.
- **Hyperparameter Optimization:** Utilized `RandomizedSearchCV` with Stratified 5-Fold Cross Validation.
- **Explainability:** Identifies top churn drivers to inform business strategy.
- **Professional Evaluation:** Tracks precision, recall, F1, ROC-AUC, alongside visual metrics (Confusion Matrix, ROC Curve).

---

## 🏗️ Repository Structure

```text
customer_churn_prediction/
├── data/                    # Raw and processed datasets (Telco-Customer-Churn.csv)
├── notebooks/               # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                     # Core source code
│   ├── train_pro.py         # Advanced ML training pipeline
│   ├── data_preprocessing.py# Helper functions
│   └── evaluate.py          # Standalone evaluation scripts
├── outputs/                 # Automatically generated plots (ROC, Confusion Matrix, etc.)
├── model/                   # Serialized model artifacts (.pkl)
├── requirements.txt         # Project dependencies
└── README.md                # Documentation
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. Environment Setup
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Data Acquisition
Download the **Telco Customer Churn** dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or via IBM's GitHub mirror, and place it in the `data/` directory:
```bash
wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv -P data/
```

---

## 💻 Usage

To execute the full training pipeline (includes preprocessing, tuning, and plotting):
```bash
python -m src.train_pro
```

**What happens under the hood?**
1. **Preprocessing:** Null handling, one-hot encoding, and target binarization.
2. **Modeling Pipeline:** Data flows through SMOTE into the Random Forest.
3. **Tuning:** Stratified 5-Fold Randomized Search CV tests various hyperparameter sets.
4. **Serialization:** The best pipeline is saved to `model/best_rf_pipeline.pkl`.
5. **Evaluation:** Outputs metrics and generates `confusion_matrix.png`, `roc_curve.png`, and `feature_importance.png` in the `outputs/` folder.

---

## 📈 Results & Insights

*Note: Final metrics are calculated purely on an unseen 20% hold-out test set.*

* **Algorithm:** Random Forest Classifier (Tuned)
* **Class Imbalance Strategy:** Synthetic Minority Over-sampling Technique (SMOTE)

*(Insert a screenshot of your feature importance plot or ROC curve here! Example: `![ROC Curve](outputs/roc_curve.png)`)*

### Business Impact
By analyzing the **Feature Importance**, the business can target specific cohorts:
- Customers with **Month-to-Month contracts** exhibit significantly higher churn risk.
- **Tenure** is negatively correlated with churn (newer customers need better onboarding).
- **Fiber Optic** internet service correlates highly with churn, suggesting potential infrastructure or pricing issues that need addressing.


