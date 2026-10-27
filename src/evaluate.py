import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_preprocessing import load_data, preprocess, train_test_split_data

def plot_feature_importance(model, feature_names, output_path='outputs/feature_importance.png'):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load data (same preprocessing as training)
    df = load_data('data/Telco-Customer-Churn.csv')
    df = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # Load the trained model
    model = joblib.load('model/random_forest.pkl')

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('ROC-AUC:', roc_auc_score(y_test, y_proba))

    # Feature importance plot
    plot_feature_importance(model, X_train.columns)
    print('Feature importance plot saved to outputs/feature_importance.png')

if __name__ == "__main__":
    main()
