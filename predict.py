# predict.py

import pandas as pd
from src.utils import load_model
import src.config as config

# Load model
model = load_model(config.MODEL_PATH)

# Sample input
sample = pd.DataFrame([{
    "island": "Torgersen",
    "culmen_length_mm": 39.1,
    "culmen_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "MALE"
}])

# Predict
prediction = model.predict(sample)

print("Predicted species:", prediction)
