import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.data_preprocessing import load_data, preprocess, train_test_split_data

def main():
    df = load_data('data/Telco-Customer-Churn.csv')
    df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # XGBoost classifier with reasonable defaults
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Save model
    joblib.dump(clf, 'model/xgboost_classifier.pkl')

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    main()
