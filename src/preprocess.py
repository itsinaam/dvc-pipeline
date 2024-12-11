import pandas as pd

def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)
    data['Feature1'] = data['Feature1'] * 10  # Example transformation
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
