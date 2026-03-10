import pandas as pd


def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that capture suspicious transaction behavior.
    """

    # Fraud occurs mostly in these types
    df["is_transfer_or_cashout"] = (
        df["type"].isin(["TRANSFER","CASH_OUT"])
    ).astype(int)

    df["orig_balance_error"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    )

    df["dest_balance_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    # Detect full balance transfer
    df["is_full_transfer"] = (
        df["amount"] == df["oldbalanceOrg"]
    ).astype(int)

    # Account draining behavior
    df["amount_balance_ratio"] = (
        df["amount"] / (df["oldbalanceOrg"] + 1)
    )


    return df


def preprocess_dataset(df: pd.DataFrame):

    df = add_fraud_features(df)

    # Drop identifiers
    df = df.drop(columns=["nameOrig", "nameDest"])

    # One-hot encode transaction type
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    df.to_csv("data/processed_fraud_dataset2.csv", index=False)

    return X, y