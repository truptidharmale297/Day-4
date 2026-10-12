from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "dataset" / "facilities.csv"
OUTPUT_FILE = BASE_DIR / "dataset" / "cleaned_facilities.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required = [
    "facility_id", "cleanliness_score", "odor_score",
    "waste_level", "footfall", "complaints",
    "hours_since_cleaning", "hygiene_risk"
]

missing_columns = [col for col in required if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing columns in CSV: {missing_columns}")

print("Original rows:", len(df))
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().copy()

numeric_columns = [
    "cleanliness_score", "odor_score",
    "footfall", "complaints", "hours_since_cleaning"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["waste_level"] = df["waste_level"].astype("string").str.strip().str.title()
df["hygiene_risk"] = df["hygiene_risk"].astype("string").str.strip().str.title()

df = df.dropna(subset=required)

df = df[
    df["cleanliness_score"].between(0, 10)
    & df["odor_score"].between(0, 10)
    & (df["footfall"] >= 0)
    & (df["complaints"] >= 0)
    & (df["hours_since_cleaning"] >= 0)
].copy()

valid_risks = ["Low", "Medium", "High"]
df = df[df["hygiene_risk"].isin(valid_risks)].copy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nCleaned rows:", len(df))
print("Saved cleaned dataset to:", OUTPUT_FILE)
print("\nRisk label counts:")
print(df["hygiene_risk"].value_counts())