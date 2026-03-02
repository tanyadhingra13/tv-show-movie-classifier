from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier


def get_model(model_name: str, seed=42):

    if model_name == "logistic":
        return OneVsRestClassifier(
            LogisticRegression(max_iter=1000)
        )

    elif model_name == "random_forest":
        return OneVsRestClassifier(
            RandomForestClassifier(
                n_estimators=200,
                random_state=seed
            )
        )

    else:
        raise ValueError("Model not supported")
        