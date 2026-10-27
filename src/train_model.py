import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.data_preprocessing import load_data, preprocess, train_test_split_data

def main():
    # Load and preprocess data
    df = load_data('data/Telco-Customer-Churn.csv')
    df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(clf, 'model/random_forest.pkl')

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    main()
