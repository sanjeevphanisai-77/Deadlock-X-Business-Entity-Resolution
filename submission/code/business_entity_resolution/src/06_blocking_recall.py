from pathlib import Path
import pandas as pd
import re
import unicodedata
from collections import defaultdict


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"


CHUNK_SIZE = 100_000


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

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# =========================================================
# LOAD SOURCE 1
# =========================================================

print("=" * 70)
print("BLOCKING RECALL ANALYSIS")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
)

s1["business_name"] = (
    s1["business_name"].fillna("")
)

s1["business_address"] = (
    s1["business_address"].fillna("")
)

s1["name_norm"] = (
    s1["business_name"]
    .map(normalize_text)
)

s1["address_norm"] = (
    s1["business_address"]
    .map(normalize_text)
)


# =========================================================
# SOURCE 1 LOOKUPS
# =========================================================

print("Creating Source 1 lookup indexes...")

name_index = defaultdict(set)
address_index = defaultdict(set)

name_country_index = defaultdict(set)
address_country_index = defaultdict(set)


for row in s1.itertuples(index=False):

    entity_id = row.entity_id
    name = row.name_norm
    address = row.address_norm
    country = row.country

    if name:
        name_index[name].add(entity_id)

        name_country_index[
            (name, country)
        ].add(entity_id)

    if address:
        address_index[address].add(entity_id)

        address_country_index[
            (address, country)
        ].add(entity_id)


# =========================================================
# GROUND TRUTH
# =========================================================

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


# Convert ground truth into sets
ground_truth = {}

for row in gt.itertuples(index=False):

    matches = set()

    if row.matched_entity_ids:

        matches = {
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        }

    ground_truth[
        row.source1_entity_id
    ] = matches


# =========================================================
# RECALL STORAGE
# =========================================================

# For every S1 entity:
# number of true matches recovered by blocking

recovered = defaultdict(set)


# =========================================================
# ANALYZE ONE SOURCE
# =========================================================

def process_source(file_path, source_name):

    print()
    print("=" * 70)
    print(f"PROCESSING {source_name}")
    print("=" * 70)

    processed = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk["business_name"] = (
            chunk["business_name"]
            .fillna("")
        )

        chunk["business_address"] = (
            chunk["business_address"]
            .fillna("")
        )

        chunk["name_norm"] = (
            chunk["business_name"]
            .map(normalize_text)
        )

        chunk["address_norm"] = (
            chunk["business_address"]
            .map(normalize_text)
        )

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            name = row.name_norm
            address = row.address_norm
            country = row.country

            candidate_s1 = set()

            # -------------------------------------------------
            # Rule 1: exact normalized name
            # -------------------------------------------------

            if name:

                candidate_s1.update(
                    name_index.get(name, ())
                )

            # -------------------------------------------------
            # Rule 2: exact normalized address
            # -------------------------------------------------

            if address:

                candidate_s1.update(
                    address_index.get(address, ())
                )

            # -------------------------------------------------
            # Rule 3: name + country
            # -------------------------------------------------

            if name:

                candidate_s1.update(
                    name_country_index.get(
                        (name, country),
                        ()
                    )
                )

            # -------------------------------------------------
            # Rule 4: address + country
            # -------------------------------------------------

            if address:

                candidate_s1.update(
                    address_country_index.get(
                        (address, country),
                        ()
                    )
                )

            # -------------------------------------------------
            # Only retain candidates where this source record
            # is actually a ground-truth match.
            #
            # This lets us calculate recall.
            # -------------------------------------------------

            for s1_id in candidate_s1:

                if entity_id in ground_truth.get(
                    s1_id,
                    set()
                ):

                    recovered[s1_id].add(
                        entity_id
                    )

            processed += 1

        print(
            f"Processed {processed:,} {source_name} records...",
            end="\r"
        )

    print()


# =========================================================
# PROCESS S2 + S3
# =========================================================

process_source(
    S2_FILE,
    "Source 2"
)

process_source(
    S3_FILE,
    "Source 3"
)


# =========================================================
# CALCULATE ENTITY-LEVEL RECALL
# =========================================================

print()
print("=" * 70)
print("ENTITY-LEVEL BLOCKING RECALL")
print("=" * 70)


total_true_matches = 0
total_recovered_matches = 0

entities_with_true_matches = 0
entities_with_all_matches_recovered = 0
entities_with_partial_recall = 0
entities_with_zero_recall = 0


recall_values = []


for s1_id, true_matches in ground_truth.items():

    if not true_matches:
        continue

    entities_with_true_matches += 1

    recovered_matches = recovered.get(
        s1_id,
        set()
    )

    true_count = len(true_matches)
    recovered_count = len(
        true_matches & recovered_matches
    )

    total_true_matches += true_count
    total_recovered_matches += recovered_count

    recall = recovered_count / true_count

    recall_values.append(recall)

    if recovered_count == 0:

        entities_with_zero_recall += 1

    elif recovered_count == true_count:

        entities_with_all_matches_recovered += 1

    else:

        entities_with_partial_recall += 1


# =========================================================
# RESULTS
# =========================================================

overall_recall = (
    total_recovered_matches /
    total_true_matches
)

average_entity_recall = sum(
    recall_values
) / len(recall_values)


print(
    f"\nEntities with true matches: "
    f"{entities_with_true_matches:,}"
)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)

print(
    f"Recovered true matches: "
    f"{total_recovered_matches:,}"
)

print(
    f"Overall match recall: "
    f"{overall_recall * 100:.4f}%"
)

print(
    f"Average entity-level recall: "
    f"{average_entity_recall * 100:.4f}%"
)

print(
    f"\nEntities with ZERO recall: "
    f"{entities_with_zero_recall:,}"
)

print(
    f"Entities with PARTIAL recall: "
    f"{entities_with_partial_recall:,}"
)

print(
    f"Entities with ALL matches recovered: "
    f"{entities_with_all_matches_recovered:,}"
)


# =========================================================
# RECALL DISTRIBUTION
# =========================================================

import numpy as np

recall_array = np.array(
    recall_values
)

print("\nEntity recall percentiles:")

for percentile in [0, 10, 25, 50, 75, 90, 95, 99, 100]:

    value = np.percentile(
        recall_array,
        percentile
    )

    print(
        f"{percentile:>3}th percentile: "
        f"{value * 100:7.3f}%"
    )


print()
print("=" * 70)
print("BLOCKING RECALL ANALYSIS COMPLETE")
print("=" * 70)