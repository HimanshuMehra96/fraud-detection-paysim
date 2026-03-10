import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from features import preprocess_dataset


DATA_PATH = "data/fraud_detection_paysim_dataset.csv"

def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Preprocessing data...")

    X, y = preprocess_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=0.2,
        random_state=42
    )


    # Save test set
    X_test.to_csv("data/processed_X_test.csv", index=False)
    y_test.to_csv("data/processed_y_test.csv", index=False)

    print("Training model...")

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    print(scale_pos_weight)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Saving model...")

    model.save_model("models/xgb_model2.json")

    print("Training complete.")

    print(classification_report(y_test, model.predict(X_test)))


if __name__ == "__main__":
    main()