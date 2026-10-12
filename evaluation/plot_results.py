from pathlib import Path
import json
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_FILE = BASE_DIR / "evaluation" / "model_comparison.json"
OUTPUT_FILE = BASE_DIR / "evaluation" / "model_accuracy.png"

if not RESULT_FILE.exists():
    print("Result file not found.")
    print("Run models/train_model.py first.")
    raise SystemExit(1)

with open(RESULT_FILE, "r", encoding="utf-8") as file:
    results = json.load(file)

model_names = list(results.keys())
accuracies = [
    results[name]["accuracy"] * 100
    for name in model_names
]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(OUTPUT_FILE)
plt.show()

print("Chart saved to:", OUTPUT_FILE)