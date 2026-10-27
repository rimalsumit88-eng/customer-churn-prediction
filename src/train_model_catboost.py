import joblib
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.data_preprocessing import load_data, preprocess, train_test_split_data

def main():
    df = load_data('data/Telco-Customer-Churn.csv')
    df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # CatBoost can handle categorical features, but after one-hot encoding we treat all as numeric
    model = CatBoostClassifier(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function='Logloss',
        eval_metric='AUC',
        verbose=False,
        random_seed=42,
        thread_count=-1,
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Save model
    joblib.dump(model, 'model/catboost_classifier.pkl')

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    main()
