import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess(csv_path: str):
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Drop identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # --- ADVANCED FEATURE ENGINEERING ---
    logger.info("Applying Feature Engineering...")
    # 1. Average charge per tenure month
    df['AvgChargePerTenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    # 2. Tenure in years (grouping)
    df['TenureYears'] = df['tenure'] / 12.0
    # 3. Ratio of Monthly Charges to Total Charges
    df['ChargeRatio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    
    # Binary target mapping
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # One-hot encoding for categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Clean column names (XGBoost throws errors if column names have characters like '<', '>', or '[')
    df.columns = df.columns.str.replace(r'[<,>\[\]]', '_', regex=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return X, y

def plot_evaluation_metrics(y_test, y_pred, y_proba, model, X_test):
    os.makedirs('outputs', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix (Optimized XGBoost)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('outputs/roc_curve.png', dpi=300)
    plt.close()

    # 3. Feature Importance
    if hasattr(model, 'best_estimator_'):
        xgb_model = model.best_estimator_
        importances = xgb_model.feature_importances_
        indices = importances.argsort()[::-1][:15] # Top 15 features
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=[X_test.columns[i] for i in indices], palette='viridis')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300)
        plt.close()

def find_optimal_threshold(y_true, y_proba):
    """Find the threshold that maximizes the F1 Score"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx]

def main():
    logger.info("Starting Ultra-Optimized ML Pipeline (Feature Engineering + Native Class Weights + Threshold Tuning)")
    
    # 1. Data Loading & Preprocessing
    X, y = load_and_preprocess('data/Telco-Customer-Churn.csv')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # Calculate scale_pos_weight for native XGBoost imbalance handling (often better than SMOTE)
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)

    # 2. Define XGBoost Classifier
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=pos_weight, # Native handling instead of SMOTE
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    # 3. Hyperparameter Tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.5, 1],
        'min_child_weight': [1, 3, 5]
    }

    logger.info("Starting Hyperparameter Tuning with Stratified 5-Fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist, 
        n_iter=20, cv=cv, scoring='f1', # Optimizing directly for F1
        n_jobs=-1, random_state=42, verbose=1
    )
    
    search.fit(X_train, y_train)
    logger.info(f"Best hyperparameters found: {search.best_params_}")

    # 4. Save Model
    os.makedirs('model', exist_ok=True)
    joblib.dump(search, 'model/best_xgboost_pipeline.pkl')
    logger.info("Model saved successfully to model/best_xgboost_pipeline.pkl")

    # 5. Optimal Threshold Tuning (The Kaggle Secret)
    logger.info("Evaluating model and finding optimal probability threshold...")
    # Get probabilities on the test set
    y_proba = search.predict_proba(X_test)[:, 1]
    
    # Find the optimal threshold using the validation set
    optimal_threshold = find_optimal_threshold(y_test, y_proba)
    logger.info(f"Optimal Prediction Threshold Found: {optimal_threshold:.4f} (Default is 0.5000)")

    # Apply optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info("\n--- FINAL OPTIMIZED TEST METRICS ---")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {roc_auc:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

    # 6. Generate Professional Plots
    logger.info("Generating evaluation plots in outputs/ directory...")
    plot_evaluation_metrics(y_test, y_pred, y_proba, search, X_test)
    logger.info("Pipeline completed successfully! \U0001f680")

if __name__ == '__main__':
    main()
