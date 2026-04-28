import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def mitigate_bias(df, target, sensitive):
    min_count = df[sensitive].value_counts().min()
    balanced = df.groupby(sensitive).sample(min_count)

    X = balanced.drop(columns=[target])
    y = balanced[target]

    X = pd.get_dummies(X, drop_first=True)
    sens = balanced[sensitive]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sens, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc_dict = {}
    for g in sens_test.unique():
        idx = sens_test == g
        acc_dict[str(g)] = accuracy_score(y_test[idx], y_pred[idx])

    values = list(acc_dict.values())
    gap = abs(values[0] - values[1]) if len(values) == 2 else 0

    return acc_dict, gap