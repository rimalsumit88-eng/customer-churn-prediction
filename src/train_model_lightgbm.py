import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.data_preprocessing import load_data, preprocess, train_test_split_data

def main():
    df = load_data('data/Telco-Customer-Churn.csv')
    df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # LightGBM dataset objects
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'verbosity': -1,
        'seed': 42,
    }

    # Train without early stopping (unsupported in this version) and suppress verbose output
    clf = lgb.train(params, train_data, num_boost_round=300, valid_sets=[valid_data], verbose_eval=False)

    # Save model
    joblib.dump(clf, 'model/lightgbm_classifier.pkl')

    # Predict
    # Predictions and evaluation
    y_proba = clf.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))

if __name__ == '__main__':
    main()
