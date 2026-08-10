from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEW_FILE = PROJECT_ROOT / "data" / "processed" / "log_returns.csv"
REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "nasdaq100_log_returns_clean.csv"
)


new = pd.read_csv(NEW_FILE, index_col=0)
reference = pd.read_csv(REFERENCE_FILE, index_col=0)


print("\n=== LOG RETURN VALIDATION ===")

print(f"New shape:       {new.shape}")
print(f"Reference shape: {reference.shape}")

same_columns = list(new.columns) == list(reference.columns)
same_index = list(new.index) == list(reference.index)

print(f"Same column order: {same_columns}")
print(f"Same dates:        {same_index}")


if new.shape == reference.shape:
    difference = np.abs(new.to_numpy() - reference.to_numpy())

    print(f"Maximum absolute difference: {difference.max():.12g}")
    print(f"Mean absolute difference:    {difference.mean():.12g}")

    equivalent = np.allclose(
        new.to_numpy(),
        reference.to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )

    print(f"Numerically equivalent: {equivalent}")
else:
    print("Cannot compare values because shapes differ.")