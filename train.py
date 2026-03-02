import argparse
import os
import joblib
from sklearn.pipeline import Pipeline
from data_loader import load_data
from preprocess import preprocess_data
from model import get_model


def train(data_path: str, model_name: str, seed: int):

    # Load data
    df = load_data(data_path)

    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor, mlb = preprocess_data(
        df, "listed_in", seed=seed
    )

    print("Data preprocessing completed. - train.py:20")

    # Get model
    model = get_model(model_name, seed)

    # Create pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    # Save model and mlb separately per model
    model_filename = f"outputs/model_{model_name}.pkl"
    mlb_filename = f"outputs/mlb_{model_name}.pkl"

    joblib.dump(pipeline, model_filename)
    joblib.dump(mlb, mlb_filename)

    print(f"Model training completed. - train.py:44")
    print(f"Saved model to {model_filename} - train.py:45")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to dataset CSV")
    parser.add_argument("--model_name", required=True, help="Model name: logistic or random_forest")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    train(args.data_path, args.model_name, args.seed)