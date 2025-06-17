import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def handle_missing(df):
    """
    Handle missing values:
    - Drop columns with too many missing values
    - Impute numeric with median
    - Impute categorical with mode
    """
    df = df.copy()
    # Drop columns with >80% missing
    # we have already cleaned teh data in task-1 but lets check if any
    threshold = 0.8 * len(df)
    df = df.loc[:, df.isnull().sum() < threshold]

    # Impute numeric and categorical separately
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns

    for col in num_cols:
        if df[col].isnull().sum():
            df[col].fillna(df[col].median(), inplace=True)

    for col in cat_cols:
        if df[col].isnull().sum():
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def engineer_features(df):
    """
    Create new features:
    - VehicleAge
    - ClaimRatio
    """
    df = df.copy()

    if "TransactionMonth" in df.columns:
        df["TransactionYear"] = df["TransactionMonth"].dt.year
    else:
        df["TransactionYear"] = 2015  # fallback default

    df["VehicleAge"] = df["TransactionYear"] - df["RegistrationYear"]
    df["VehicleAge"] = df["VehicleAge"].clip(lower=0)

    df["ClaimRatio"] = np.where(
        df["TotalPremium"] > 0,
        df["TotalClaims"] / df["TotalPremium"],
        0
    )

    return df


def encode_categoricals(df, exclude=[]):
    """
    One-hot encode categorical variables.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [col for col in cat_cols if col not in exclude]

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def filter_claims_positive(df):
    """
    Keep only rows with positive TotalClaims (for severity modeling).
    """
    return df[df["TotalClaims"] > 0].copy()

def train_test_split_df(df, target_col, test_size=0.2, random_state=42):
    """
    Split data into X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def full_preprocess_pipeline(df, target_col, severity_only=True):
    """
    Run entire preprocessing pipeline.
    """
    df = handle_missing(df)
    df = engineer_features(df)

    if severity_only:
        df = filter_claims_positive(df)

    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test = train_test_split_df(df, target_col)
    return X_train, X_test, y_train, y_test, df
