import argparse
import os
import joblib
import json
from sklearn.metrics import f1_score, hamming_loss
from data_loader import load_data
from preprocess import preprocess_data


def evaluate(model_path: str, data_path: str, model_name: str):

    print("Loading model... - evaluate.py:12")
    pipeline = joblib.load(model_path)

    print("Loading mlb... - evaluate.py:15")
    mlb = joblib.load(f"outputs/mlb_{model_name}.pkl")

    print("Loading data... - evaluate.py:18")
    df = load_data(data_path)

    print("Preprocessing... - evaluate.py:21")
    X_train, X_test, y_train, y_test, _, _ = preprocess_data(df, "listed_in")

    print("Predicting... - evaluate.py:24")
    y_pred = pipeline.predict(X_test)

    print("Calculating metrics... - evaluate.py:27")

    metrics = {
        "f1_micro": f1_score(y_test, y_pred, average="micro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "hamming_loss": hamming_loss(y_test, y_pred)
    }

    os.makedirs("outputs", exist_ok=True)

    metrics_filename = f"outputs/metrics_{model_name}.json"

    with open(metrics_filename, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation completed. - evaluate.py:42")
    print(f"Metrics saved to {metrics_filename} - evaluate.py:43")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", required=True)

    args = parser.parse_args()

    evaluate(args.model_path, args.data_path, args.model_name)