from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
import pandas as pd


def preprocess_data(df: pd.DataFrame, target_column: str, test_size=0.2, seed=42):

    # Drop rows with missing target
    df = df.dropna(subset=[target_column])

    # Convert genre column into list
    df[target_column] = df[target_column].apply(lambda x: x.split(", "))

    # Multi-label binarization
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df[target_column])

    X = df.drop(columns=[target_column])

    categorical_cols = ["type", "rating", "country", "platform"]
    numerical_cols = ["release_year"]

    X = X.drop(
        columns=["title", "description", "cast", "director", "date_added", "duration"],
        errors="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return X_train, X_test, y_train, y_test, preprocessor, mlb