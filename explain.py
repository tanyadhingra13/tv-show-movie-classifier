import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os


def feature_importance(model_path: str, model_name: str):

    pipeline = joblib.load(model_path)
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessing"]

    # Get feature names
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = None

    # Only proceed if OneVsRestClassifier
    if hasattr(model, "estimators_"):

        # Collect only valid estimators (skip _ConstantPredictor)
        valid_importances = [
            est.feature_importances_
            for est in model.estimators_
            if hasattr(est, "feature_importances_")
        ]

        if len(valid_importances) == 0:
            print("No valid feature importances available. - explain.py:31")
            return

        # Average feature importances across labels
        importances = np.mean(valid_importances, axis=0)

        # Get top 20 features
        indices = np.argsort(importances)[-20:]

        labels = (
            [feature_names[i] for i in indices]
            if feature_names is not None
            else indices
        )

        os.makedirs("outputs", exist_ok=True)

        filename = f"outputs/feature_importance_{model_name}.png"

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), labels)
        plt.title(f"Top 20 Feature Importances ({model_name})")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"Feature importance saved to {filename} - explain.py:58")

    else:
        print("Feature importance not available for this model. - explain.py:61")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)

    args = parser.parse_args()

    feature_importance(args.model_path, args.model_name)