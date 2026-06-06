# predict.py

import pandas as pd
from src.utils import load_model
import src.config as config

# Load model
model = load_model(config.MODEL_PATH)

# Sample input
sample = pd.DataFrame([{
    "island": "Torgersen",
    "culmen_length_mm": 42.1,
    "culmen_depth_mm": 16.7,
    "flipper_length_mm": 171,
    "body_mass_g": 3760,
    "sex": "MALE"
}])

# Predict
prediction = model.predict(sample)

print("Predicted species:", prediction)
