# src/evaluate.py

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)


def main():

    print("Loading test dataset...")

    X_test = pd.read_csv("data/processed_X_test.csv")
    y_test = pd.read_csv("data/processed_y_test.csv").values.ravel()

    print("Loading trained model...")

    model = xgb.XGBClassifier()
    model.load_model("models/xgb_model.json")

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    cm = confusion_matrix(y_test, preds)

    print("Confusion Matrix")
    print(cm)

    pr_auc = average_precision_score(y_test, probs)

    print("PR-AUC:", pr_auc)

    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.figure()

    plt.plot(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.savefig("results/pr_curve.png")

    print("Saved PR curve to results/pr_curve.png")


if __name__ == "__main__":
    main()