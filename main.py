# main.py

from src import config
from src.data import load_data
from src.model import build_model
from src.train import train
from src.evaluate import evaluate
from src.utils import save_model

from sklearn.model_selection import cross_val_score

# Load data
X_train, X_test, y_train, y_test = load_data(
    config.DATA_PATH,
    config.TEST_SIZE,
    config.RANDOM_STATE
)

# Build model
model = build_model(
    config.MAX_DEPTH,
    config.MIN_SAMPLES_SPLIT
)

# Train model
model = train(model, X_train, y_train)

# -------------------------
# Evaluation (hold-out test)
# -------------------------
acc, report = evaluate(model, X_test, y_test)

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

print("\nHold-out Accuracy:", acc)
print("\nReport:\n", report)

# -------------------------
# Cross Validation (IMPORTANT)
# -------------------------
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Save model
save_model(model, config.MODEL_PATH)

print("\nModel saved successfully!")
