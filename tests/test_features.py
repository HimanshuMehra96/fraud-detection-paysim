import pandas as pd
import sys
import os

sys.path.append(os.path.abspath("src"))

from features import add_fraud_features, preprocess_dataset


def create_sample_df():
    """Create a small synthetic dataset for testing."""
    data = {
        "type": ["TRANSFER", "PAYMENT"],
        "amount": [100, 50],
        "oldbalanceOrg": [100, 200],
        "newbalanceOrig": [0, 150],
        "oldbalanceDest": [0, 0],
        "newbalanceDest": [100, 50],
        "nameOrig": ["C1", "C2"],
        "nameDest": ["C3", "C4"],
        "isFraud": [1, 0],
    }

    return pd.DataFrame(data)


def test_add_fraud_features_columns():
    """Check if new engineered features are added."""
    df = create_sample_df()

    df = add_fraud_features(df)

    expected_columns = [
        "is_transfer_or_cashout",
        "orig_balance_error",
        "dest_balance_error",
        "is_full_transfer",
        "amount_balance_ratio",
    ]

    for col in expected_columns:
        assert col in df.columns


def test_full_transfer_detection():
    """Verify full transfer logic."""
    df = create_sample_df()

    df = add_fraud_features(df)

    # First row amount == oldbalanceOrg
    assert df.loc[0, "is_full_transfer"] == 1

    # Second row not full transfer
    assert df.loc[1, "is_full_transfer"] == 0


def test_preprocess_dataset_output():
    """Check preprocessing returns X and y correctly."""
    df = create_sample_df()

    X, y = preprocess_dataset(df)

    # Check target separated
    assert "isFraud" not in X.columns
    assert len(X) == len(y)

    # Ensure identifiers dropped
    assert "nameOrig" not in X.columns
    assert "nameDest" not in X.columns