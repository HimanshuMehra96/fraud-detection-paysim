# Fraud Detection using PaySim Dataset
## Overview

This project builds a machine learning pipeline to detect fraudulent financial transactions using the PaySim dataset. The goal is to identify fraud in a highly imbalanced dataset while maintaining high recall to minimize missed fraudulent activity.

The project implements a complete ML workflow including:

1. Exploratory data analysis (EDA)

2. Fraud pattern discovery

3. Feature engineering

4. Model training using gradient boosted trees

5. Evaluation using precision–recall metrics

The final model achieves near-perfect performance by combining domain-inspired feature engineering with gradient boosting.

## Dataset

The dataset used is the PaySim synthetic mobile money transactions dataset.

Dataset source:
https://www.kaggle.com/datasets/ealaxi/paysim1

Source: Kaggle
License: CC BY-SA 4.0 - Link: https://creativecommons.org/licenses/by-sa/4.0/

| Property | Value |
|----------|-------|
| Samples | ~6.3 million transactions |
| Original features | 10 |
| Fraud rate | ~0.13% |
| Task | Binary classification |

Important features include:

1. type – transaction type

2. amount

3. oldbalanceOrg

4. newbalanceOrig

5. oldbalanceDest

6. newbalanceDest

7. isFraud

The dataset is extremely imbalanced, making precision–recall metrics more suitable than accuracy.

## Project Structure
```text
fraud-detection-paysim
│
├── data
│   ├── PaySim dataset (downloaded from Kaggle)
│   ├── X_test.csv
│   └── y_test.csv
│
├── notebooks
│   ├── EDA
│   ├── feature analysis
│   └── baseline experiments
│
├── src
│   ├── features.py      # feature engineering pipeline
│   ├── train.py         # model training script
│   └── evaluate.py      # evaluation script
│
├── models
│   └── xgb_model.json
│
├── results
│   └── pr_curve.png
│
├── requirements.txt
├── environment.yml
└── README.md
```

## Installation

Clone the repository:

git clone https://github.com/HimanshuMehra96/fraud-detection-paysim.git
```bash
cd fraud-detection-paysim
```

Create the environment using Conda:

```bash
conda env create -f environment.yml
conda activate fraud-detection
```

Alternatively install using pip:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the PaySim dataset from Kaggle:

**https://www.kaggle.com/datasets/ealaxi/paysim1**

Place the CSV file inside:

```text
data/
```

Rename the file to

```text
fraud_detection_paysim_dataset.csv
```

Example:

```text
data/fraud_detection_paysim_dataset.csv
```

## Feature Engineering

Several engineered features capture suspicious transaction patterns.

### Balance inconsistency detection

Fraudulent transactions often violate balance conservation.

```text
orig_balance_error =
oldbalanceOrg - newbalanceOrig - amount
```
```text
dest_balance_error =
oldbalanceDest + amount - newbalanceDest
```

### Account draining behavior

Fraud frequently involves transferring nearly the entire balance.

```text
amount_balance_ratio =
amount / oldbalanceOrg
```

### Transaction type filtering

Fraud occurs almost exclusively in:

```text
TRANSFER

CASH_OUT
```

Binary indicator:
```text
is_transfer_or_cashout
```

## Model

The final model uses gradient boosted trees with XGBoost.

Model parameters:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.1
scale_pos_weight = class imbalance ratio
```

XGBoost was chosen due to its strong performance on structured/tabular datasets and its ability to model nonlinear feature interactions.

## Training

Run the training pipeline:

```bash
python src/train.py
```

This will:

1. Load the dataset

2. Generate engineered features

3. Split train/test data

4. Train the model

5. Save the trained model to
```text
models/xgb_model.json
```

## Evaluation

Run model evaluation:

```bash
python src/evaluate.py
```

This will:

1. Load the trained model

2. Evaluate performance on the saved test set

3. Compute metrics

4. Generate a precision–recall curve

Saved results:

```text
results/pr_curve.png
```

## Model Performance

Confusion Matrix:

```text
[[1270876       5]
 [      4    1639]]
```

Performance metrics:

| Metric | Value |
|------|------|
| True Positives | 1639 |
| False Positives | 5 |
| False Negatives | 4 |
| True Negatives | 1,270,876 |
| Precision | 0.997 |
| Recall | 0.997 |
| PR-AUC | ~0.998 |

The model detects nearly all fraudulent transactions while maintaining extremely low false positives.

## Unit Testing

```bash
pytest tests/
```

## Key Insights from EDA

1. Several patterns strongly correlate with fraud:

2. Fraud occurs almost exclusively in TRANSFER and CASH_OUT transactions.

3. Fraud often drains the entire origin account balance.

4. Fraud transactions frequently introduce balance inconsistencies.

5. Feature engineering based on these patterns dramatically improved model performance.

## Limitations

The PaySim dataset is synthetic, meaning patterns may not perfectly represent real-world fraud scenarios. Real banking data is typically noisier and less deterministic.
