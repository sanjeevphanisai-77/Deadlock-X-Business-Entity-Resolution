from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[4]

GROUND_TRUTH = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VALIDATION_FRACTION = 0.10
RANDOM_SEED = 42


# ---------------------------------------------------------
# Load Source-1 ground truth
# ---------------------------------------------------------

print("Loading ground truth...")

df = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

print(f"Total entities: {len(df):,}")


# ---------------------------------------------------------
# Shuffle Source-1 entities
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

indices = np.arange(len(df))
rng.shuffle(indices)

validation_size = int(
    len(df) * VALIDATION_FRACTION
)

validation_indices = indices[:validation_size]
training_indices = indices[validation_size:]


train_df = df.iloc[training_indices].copy()
validation_df = df.iloc[validation_indices].copy()


# ---------------------------------------------------------
# Save splits
# ---------------------------------------------------------

train_path = OUTPUT_DIR / "train_ground_truth.tsv"
validation_path = OUTPUT_DIR / "validation_ground_truth.tsv"

train_df.to_csv(
    train_path,
    sep="\t",
    index=False
)

validation_df.to_csv(
    validation_path,
    sep="\t",
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

print(f"Training entities   : {len(train_df):,}")
print(f"Validation entities : {len(validation_df):,}")

print()
print(f"Training file   : {train_path}")
print(f"Validation file : {validation_path}")

print("=" * 70)