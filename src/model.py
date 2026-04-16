# src/model.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def build_model(max_depth, min_samples_split):

    categorical_features = ["island", "sex"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocess", preprocessor),
        ("dt", DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        ))
    ])

    return model
