# src/data.py

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path, test_size, random_state):
    df = pd.read_csv(path)

    # Handle missing values
    df = df.replace("NA", pd.NA)
    df = df.dropna()

    # Features & target
    X = df.drop("species", axis=1)
    y = df["species"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
