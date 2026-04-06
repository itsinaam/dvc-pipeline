import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

"""train.py

DVC Pipeline Stage: Train
========================

This script is intended to be used as a DVC stage that trains a machine learning
model from a processed dataset and produces two tracked artifacts:

- A serialized model file (e.g., `models/model.joblib`)
- A metrics file (e.g., `metrics.txt`) that DVC can read and compare across runs

Typical DVC usage (conceptual)
------------------------------
- Input artifact:   data/processed.csv
- Output artifacts: models/model.joblib, metrics.txt

The stage is usually executed by DVC via a command similar to:
    python src/train.py --input data/processed.csv --model models/model.joblib --metrics metrics.txt

What this stage does
--------------------
1) Loads the processed dataset CSV.
2) Splits the data into train/test sets.
3) Trains a RandomForestClassifier.
4) Evaluates accuracy on the test set.
5) Writes the accuracy to a plain-text metrics file.
6) Saves the trained model using joblib.

Dataset expectations
--------------------
The input CSV is expected to contain:
- Feature columns: `Feature1`, `Feature2`
- Target column:   `Target`

Notes for clients / maintainers
-------------------------------
- This is a minimal example model. Replace the algorithm, features, metrics,
  and hyperparameters to match your real use case.
- The metrics file is written as simple key/value text ("accuracy: <value>")
  to keep it easy to parse and review.
"""


def train(input_path, model_path, metrics_path):
    """Train a model and write metrics.

    Parameters
    ----------
    input_path : str
        Path to the processed dataset CSV.
    model_path : str
        Path where the trained model will be saved (joblib format).
    metrics_path : str
        Path where the evaluation metrics will be written.

    Returns
    -------
    None
        Writes the model to `model_path` and metrics to `metrics_path`.
    """
    data = pd.read_csv(input_path)

    # Define features and target
    X = data[['Feature1', 'Feature2']]
    y = data['Target']

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Write metrics in a simple, DVC-friendly format
    with open(metrics_path, 'w') as f:
        f.write(f"accuracy: {accuracy}\n")

    # Persist model artifact
    joblib.dump(model, model_path)


if __name__ == "__main__":
    """CLI entrypoint.

    DVC typically calls this script with `--input`, `--model`, and `--metrics`.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train a model for the DVC pipeline")
    parser.add_argument('--input', required=True, help='Path to processed CSV')
    parser.add_argument('--model', required=True, help='Path to output model file (joblib)')
    parser.add_argument('--metrics', required=True, help='Path to output metrics text file')
    args = parser.parse_args()

    train(args.input, args.model, args.metrics)
