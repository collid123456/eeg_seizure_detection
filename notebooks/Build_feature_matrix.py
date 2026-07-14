"""Build a feature matrix from all Bonn signals and check class separation."""

import sys

sys.path.append("src")

import pandas as pd
import matplotlib.pyplot as plt

from data.load_bonn import load_bonn_dataset
from preprocessing.filters import preprocess_signal
from features.extract import extract_features

# Load all signals
signals, labels = load_bonn_dataset()

# Filter and extract features for every signal
rows = []
for signal, label in zip(signals, labels):
    filtered = preprocess_signal(signal)
    features = extract_features(filtered)
    features["label"] = label
    rows.append(features)

# Build the dataframe
df = pd.DataFrame(rows)
print(f"Feature matrix shape: {df.shape}")
print(df.head())

# Save it for later use
df.to_csv("data/processed/bonn_features.csv", index=False)
print("\nSaved feature matrix to data/processed/bonn_features.csv")

# Quick visual check: does line_length separate the classes?
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for label, name, color in [(0, "Non-seizure", "steelblue"), (1, "Seizure", "crimson")]:
    subset = df[df["label"] == label]
    axes[0].hist(subset["line_length"], bins=20, alpha=0.6, label=name, color=color)
    axes[1].hist(subset["rms"], bins=20, alpha=0.6, label=name, color=color)

axes[0].set_title("Line Length by Class")
axes[0].set_xlabel("Line Length")
axes[0].legend()

axes[1].set_title("RMS by Class")
axes[1].set_xlabel("RMS")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/figures/feature_separation.png", dpi=150)
print("Saved plot to reports/figures/feature_separation.png")
plt.show()
