from sklearn.metrics import accuracy_score

def detect_bias(y_test, y_pred, sens_test):
    overall_acc = accuracy_score(y_test, y_pred)

    acc_dict = {}
    for g in sens_test.unique():
        idx = sens_test == g
        acc_dict[str(g)] = accuracy_score(y_test[idx], y_pred[idx])

    values = list(acc_dict.values())
    bias_gap = abs(values[0] - values[1]) if len(values) == 2 else 0

    return overall_acc, acc_dict, bias_gap