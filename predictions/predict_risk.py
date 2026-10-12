from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "logistic_regression.joblib"
OUTPUT_FILE = BASE_DIR / "predictions" / "facility_predictions.csv"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Model not found. Run models/train_model.py first."
    )

model = joblib.load(MODEL_FILE)

# Example facility to predict
new_facility = pd.DataFrame([{
    "cleanliness_score": 4,
    "odor_score": 8,
    "waste_level": "High",
    "complaints": 12,
    "footfall": 300,
    "hours_since_cleaning": 14
}])

risk = model.predict(new_facility)[0]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

new_facility["predicted_hygiene_risk"] = risk
new_facility.to_csv(OUTPUT_FILE, index=False)

print("Predicted hygiene risk:", risk)
print("Prediction saved to:", OUTPUT_FILE)