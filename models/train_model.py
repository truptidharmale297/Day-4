from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "dataset" / "cleaned_facilities.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_FILE = BASE_DIR / "evaluation" / "model_comparison.json"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run clean_data.py first."
    )

df = pd.read_csv(DATA_FILE)

features = [
    "cleanliness_score",
    "odor_score",
    "waste_level",
    "complaints",
    "footfall",
    "hours_since_cleaning"
]
target = "hygiene_risk"

X = df[features]
y = df[target]

if len(df) < 10 or y.nunique() < 2:
    raise ValueError(
        "Please provide at least 10 valid rows and at least 2 risk classes."
    )

# Stratify only if every class has enough rows for the split.
stratify_values = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=stratify_values
)

numeric_features = [
    "cleanliness_score",
    "odor_score",
    "complaints",
    "footfall",
    "hours_since_cleaning"
]
categorical_features = ["waste_level"]

def make_preprocessor():
    return ColumnTransformer([
        ("numbers", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("categories", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])

algorithms = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(
        max_depth=3, random_state=42
    )
}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

results = {}

for name, algorithm in algorithms.items():
    model = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("classifier", algorithm)
    ])

    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)

    joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    results[name] = {
        "accuracy": round(float(accuracy), 4),
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }

    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.2%}")

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nComparison report saved:", REPORT_FILE)
print("Both models have been trained.")