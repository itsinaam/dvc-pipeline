import pandas as pd

"""preprocess.py

DVC Pipeline Stage: Preprocess
=============================

This script is intended to be used as a DVC stage that reads a raw dataset CSV,
applies a simple feature transformation, and writes the processed dataset CSV.

Typical DVC usage (conceptual)
------------------------------
- Input artifact:  data/raw.csv
- Output artifact: data/processed.csv

The stage is usually executed by DVC via a command similar to:
    python src/preprocess.py --input data/raw.csv --output data/processed.csv

What this stage does
--------------------
1) Loads the input CSV into a pandas DataFrame.
2) Applies an example transformation:
   - Multiplies the `Feature1` column by 10.
3) Saves the transformed DataFrame to the output CSV.

Notes for clients / maintainers
-------------------------------
- This is a minimal example. Replace the transformation logic with your real
  preprocessing steps (missing value handling, encoding, scaling, etc.).
- The script expects the input CSV to contain a column named `Feature1`.
"""


def preprocess(input_path, output_path):
    """Run preprocessing on a CSV dataset.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file (raw dataset).
    output_path : str
        Path where the processed CSV will be written.

    Returns
    -------
    None
        Writes the processed dataset to `output_path`.
    """
    data = pd.read_csv(input_path)

    # Example transformation (replace with real preprocessing logic)
    data['Feature1'] = data['Feature1'] * 10

    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    """CLI entrypoint.

    DVC typically calls this script with `--input` and `--output` arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess raw data for the DVC pipeline")
    parser.add_argument('--input', required=True, help='Path to input raw CSV')
    parser.add_argument('--output', required=True, help='Path to output processed CSV')
    args = parser.parse_args()

    preprocess(args.input, args.output)
