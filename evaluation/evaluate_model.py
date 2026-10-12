from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_facilities.csv"
MODEL_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_FILE)

features = [
    "cleanliness_score", "odor_score", "waste_level",
    "complaints", "footfall", "hours_since_cleaning"
]
X = df[features]
y = df["hygiene_risk"]

stratify_values = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42,
    stratify=stratify_values
)

for name in ["logistic_regression", "decision_tree"]:
    model_file = MODEL_DIR / f"{name}.joblib"

    if not model_file.exists():
        print(f"Missing {model_file.name}. Run train_model.py first.")
        continue

    model = joblib.load(model_file)
    predictions = model.predict(X_test)

    print(f"\n===== {name.upper()} =====")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Precision:", round(
        precision_score(y_test, predictions, average="macro",
                        zero_division=0), 3
    ))
    print("Recall:", round(
        recall_score(y_test, predictions, average="macro",
                     zero_division=0), 3
    ))
    print("F1 score:", round(
        f1_score(y_test, predictions, average="macro",
                 zero_division=0), 3
    ))

    labels = ["High", "Low", "Medium"]
    print("\nConfusion matrix (labels: High, Low, Medium):")
    print(confusion_matrix(y_test, predictions, labels=labels))

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))