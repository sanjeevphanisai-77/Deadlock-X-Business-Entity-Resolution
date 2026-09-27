from pathlib import Path
import pandas as pd


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


# ---------------------------------------------------------
# Load ground truth
# ---------------------------------------------------------

print("=" * 70)
print("GROUND TRUTH MATCH DISTRIBUTION")
print("=" * 70)

df = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

print(f"Total Source-1 entities: {len(df):,}")


# ---------------------------------------------------------
# Convert matched IDs into match counts
# ---------------------------------------------------------

def count_matches(value):
    if pd.isna(value) or str(value).strip() == "":
        return 0

    return len(
        [
            x
            for x in str(value).split(",")
            if x.strip()
        ]
    )


df["match_count"] = df["matched_entity_ids"].apply(count_matches)


# ---------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------

print("\nMatch count statistics:")
print(df["match_count"].describe())


# ---------------------------------------------------------
# Distribution
# ---------------------------------------------------------

print("\nExact match-count distribution:")

distribution = (
    df["match_count"]
    .value_counts()
    .sort_index()
)

for count, number in distribution.items():
    percentage = number / len(df) * 100

    print(
        f"{count:>4} matches : "
        f"{number:>10,} entities "
        f"({percentage:6.2f}%)"
    )


# ---------------------------------------------------------
# Singleton count
# ---------------------------------------------------------

zero = (df["match_count"] == 0).sum()
one = (df["match_count"] == 1).sum()
multiple = (df["match_count"] > 1).sum()

print("\nSummary:")
print(f"No matches       : {zero:,}")
print(f"Exactly 1 match  : {one:,}")
print(f"Multiple matches : {multiple:,}")


# ---------------------------------------------------------
# Maximum
# ---------------------------------------------------------

max_matches = df["match_count"].max()

print(f"\nMaximum matches for one Source-1 entity: {max_matches}")


# ---------------------------------------------------------
# Examples of high-match entities
# ---------------------------------------------------------

print("\nExamples with many matches:")

examples = (
    df[df["match_count"] >= 5]
    .sort_values("match_count", ascending=False)
    .head(10)
)

for _, row in examples.iterrows():

    print(
        f"\n{row['source1_entity_id']} "
        f"-> {row['match_count']} matches"
    )

    print(row["matched_entity_ids"])


print("\n" + "=" * 70)
print("GROUND TRUTH ANALYSIS COMPLETE")
print("=" * 70)