import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train(input_path, model_path, metrics_path):
    data = pd.read_csv(input_path)
    X = data[['Feature1', 'Feature2']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    with open(metrics_path, 'w') as f:
        f.write(f"accuracy: {accuracy}\n")
    
    joblib.dump(model, model_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--metrics', required=True)
    args = parser.parse_args()
    train(args.input, args.model, args.metrics)
