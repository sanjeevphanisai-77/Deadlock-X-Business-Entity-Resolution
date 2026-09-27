from pathlib import Path
import pandas as pd
import re
import unicodedata
import random


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    value = value.lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# =========================================================
# LOAD DATA
# =========================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
).set_index("entity_id")


print("Loading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)


# =========================================================
# CREATE SOURCE 1 EXACT INDEXES
# =========================================================

print("Creating Source 1 indexes...")

name_index = {}
address_index = {}

for entity_id, row in s1.iterrows():

    name = normalize_text(
        row["business_name"]
    )

    address = normalize_text(
        row["business_address"]
    )

    if name:
        name_index.setdefault(
            name,
            set()
        ).add(entity_id)

    if address:
        address_index.setdefault(
            address,
            set()
        ).add(entity_id)


# =========================================================
# COLLECT TRUE MATCHES
# =========================================================

random.seed(42)

missed_examples = []


# We only need a sample
MAX_EXAMPLES = 100


print("\nSearching for missed true matches...")


# ---------------------------------------------------------
# Load S2/S3 individually
# ---------------------------------------------------------

for source_file, source_name in [
    (S2_FILE, "S2"),
    (S3_FILE, "S3")
]:

    print(f"\nProcessing {source_name}...")

    source = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    source = source.set_index("entity_id")


    for _, gt_row in gt.iterrows():

        s1_id = gt_row["source1_entity_id"]

        matches = [
            x.strip()
            for x in gt_row["matched_entity_ids"].split(",")
            if x.strip()
        ]

        if not matches:
            continue

        # Only inspect IDs belonging to this source
        source_matches = [
            x for x in matches
            if x.startswith(source_name + "-")
        ]

        if not source_matches:
            continue

        if s1_id not in s1.index:
            continue

        s1_name = normalize_text(
            s1.loc[
                s1_id,
                "business_name"
            ]
        )

        s1_address = normalize_text(
            s1.loc[
                s1_id,
                "business_address"
            ]
        )


        for entity_id in source_matches:

            if entity_id not in source.index:
                continue

            row = source.loc[entity_id]

            name = normalize_text(
                row["business_name"]
            )

            address = normalize_text(
                row["business_address"]
            )


            # ---------------------------------------------
            # Did exact blocking already find this pair?
            # ---------------------------------------------

            found_by_name = (
                s1_id in name_index.get(
                    name,
                    set()
                )
            )

            found_by_address = (
                s1_id in address_index.get(
                    address,
                    set()
                )
            )


            if found_by_name or found_by_address:
                continue


            # ---------------------------------------------
            # This is a TRUE MATCH missed by exact blocking
            # ---------------------------------------------

            missed_examples.append({

                "source": source_name,

                "s1_id": s1_id,

                "s1_name": s1.loc[
                    s1_id,
                    "business_name"
                ],

                "s2_s3_id": entity_id,

                "candidate_name": row[
                    "business_name"
                ],

                "s1_address": s1.loc[
                    s1_id,
                    "business_address"
                ],

                "candidate_address": row[
                    "business_address"
                ],

                "country": row["country"]

            })


            if len(missed_examples) >= MAX_EXAMPLES:
                break

        if len(missed_examples) >= MAX_EXAMPLES:
            break

    if len(missed_examples) >= MAX_EXAMPLES:
        break


# =========================================================
# DISPLAY RESULTS
# =========================================================

print()
print("=" * 70)
print("MISSED TRUE-MATCH EXAMPLES")
print("=" * 70)


for i, example in enumerate(
    missed_examples,
    start=1
):

    print(f"\nExample {i}")
    print("-" * 60)

    print(
        f"Source       : {example['source']}"
    )

    print(
        f"S1 ID        : {example['s1_id']}"
    )

    print(
        f"Matched ID   : {example['s2_s3_id']}"
    )

    print(
        f"S1 name      : {example['s1_name']}"
    )

    print(
        f"Candidate name: {example['candidate_name']}"
    )

    print(
        f"S1 address   : {example['s1_address']}"
    )

    print(
        f"Candidate addr: {example['candidate_address']}"
    )

    print(
        f"Country      : {example['country']}"
    )


print()
print(
    f"Examples collected: "
    f"{len(missed_examples)}"
)

print("=" * 70)