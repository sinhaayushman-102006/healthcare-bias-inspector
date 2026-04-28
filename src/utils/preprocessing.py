import pandas as pd

def preprocess_data(df, target, sensitive):
    """
    Preprocess the dataset:
    - Handle missing values
    - Separate target and sensitive column
    - Encode categorical variables
    """

    # 🔹 Drop rows with missing target
    df = df.dropna(subset=[target])

    # 🔹 Fill missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # 🔹 Separate target
    y = df[target]

    # 🔹 Separate sensitive column
    sensitive_data = df[sensitive]

    # 🔹 Drop target from features
    X = df.drop(columns=[target])

    # 🔹 One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)

    return X, y, sensitive_data