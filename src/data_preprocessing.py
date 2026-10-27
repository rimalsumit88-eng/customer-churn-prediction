import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV file."""
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing:
    - Drop customerID column
    - Convert categorical variables with one‑hot encoding
    - Map target column 'Churn' to binary (Yes/No -> 1/0)
    """
    df = df.copy()
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
    # Binary target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    # One‑hot encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Apply SMOTE to the training set
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test
