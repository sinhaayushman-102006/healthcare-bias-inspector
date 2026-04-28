from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.utils.preprocessing import preprocess_data

def train_model(df, target, sensitive):
    # Preprocess
    X, y, sens = preprocess_data(df, target, sensitive)

    # Split
    try:
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sens, test_size=0.2, random_state=42, stratify=sens
        )
    except ValueError:
        # Fallback to no stratification if sensitive attribute has single-member groups
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sens, test_size=0.2, random_state=42
        )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    overall_acc = accuracy_score(y_test, y_pred)

    return model, X_test, y_test, sens_test, y_pred, overall_acc
