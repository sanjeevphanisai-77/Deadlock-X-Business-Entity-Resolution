from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"


# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_files = {
    "Source 1": TRAIN_DIR / "train_source1.tsv",
    "Source 2": TRAIN_DIR / "train_source2.tsv",
    "Source 3": TRAIN_DIR / "train_source3.tsv",
    "Ground Truth": TRAIN_DIR / "train_ground_truth.tsv",
}

datasets = {}

for name, file_path in train_files.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("File:", file_path)

    if not file_path.exists():
        print("ERROR: File not found!")
        continue

    df = pd.read_csv(file_path, sep="\t")

    datasets[name] = df

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())


# ============================================================
# SOURCE STATISTICS
# ============================================================

for name in ["Source 1", "Source 2", "Source 3"]:

    if name not in datasets:
        continue

    df = datasets[name]

    print("\n" + "=" * 70)
    print(f"{name} STATISTICS")
    print("=" * 70)

    print("\nNumber of rows:", len(df))

    print(
        "Unique entity IDs:",
        df["entity_id"].nunique()
    )

    print(
        "Duplicate entity IDs:",
        df["entity_id"].duplicated().sum()
    )

    if "country" in df.columns:

        print("\nCountry distribution:")

        print(
            df["country"]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

if "Ground Truth" in datasets:

    gt = datasets["Ground Truth"]

    print("\n" + "=" * 70)
    print("GROUND TRUTH STATISTICS")
    print("=" * 70)

    print("\nRows:", len(gt))

    print("\nColumns:")
    print(gt.columns.tolist())

    print("\nFirst 10 rows:")
    print(gt.head(10).to_string())

    if "matched_entity_ids" in gt.columns:

        matches = (
            gt["matched_entity_ids"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        empty_matches = matches.eq("")

        print(
            "\nEntities with NO matches:",
            empty_matches.sum()
        )

        print(
            "Entities with at least one match:",
            (~empty_matches).sum()
        )


print("\n" + "=" * 70)
print("DATASET EXPLORATION COMPLETE")
print("=" * 70)