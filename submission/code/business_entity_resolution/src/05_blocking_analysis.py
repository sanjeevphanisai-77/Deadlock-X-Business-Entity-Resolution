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


# =========================================================
# CONFIGURATION
# =========================================================

CHUNK_SIZE = 100_000


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)

    value = value.lower()

    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# =========================================================
# LOAD SOURCE 1
# =========================================================

print("=" * 70)
print("BLOCKING ANALYSIS")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ]
)

print(f"Source 1 rows: {len(s1):,}")


# =========================================================
# NORMALIZE SOURCE 1
# =========================================================

print("\nNormalizing Source 1 names...")

s1["name_norm"] = (
    s1["business_name"]
    .fillna("")
    .map(normalize_text)
)

s1["address_norm"] = (
    s1["business_address"]
    .fillna("")
    .map(normalize_text)
)


# =========================================================
# CREATE SOURCE 1 BLOCKING INDEXES
# =========================================================

print("\nCreating Source 1 indexes...")

name_index = defaultdict(list)
address_index = defaultdict(list)
name_country_index = defaultdict(list)
address_country_index = defaultdict(list)


for idx, row in s1.iterrows():

    name = row["name_norm"]
    address = row["address_norm"]
    country = row["country"]

    if name:
        name_index[name].append(idx)

        name_country_index[
            (name, country)
        ].append(idx)

    if address:
        address_index[address].append(idx)

        address_country_index[
            (address, country)
        ].append(idx)


print(f"Unique normalized names: {len(name_index):,}")
print(f"Unique normalized addresses: {len(address_index):,}")


# =========================================================
# GROUND TRUTH
# =========================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = gt[
    "matched_entity_ids"
].fillna("")


# =========================================================
# CREATE GROUND-TRUTH LOOKUP
# =========================================================

print("Creating ground-truth lookup...")

ground_truth = {}

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    matches = set()

    value = row["matched_entity_ids"]

    if value:

        for entity_id in value.split(","):

            entity_id = entity_id.strip()

            if entity_id:
                matches.add(entity_id)

    ground_truth[s1_id] = matches


# =========================================================
# ANALYZE SOURCE 2 / SOURCE 3
# =========================================================

def analyze_source(
    file_path,
    source_name,
    entity_prefix
):

    print()
    print("=" * 70)
    print(f"ANALYZING {source_name}")
    print("=" * 70)

    total_rows = 0

    exact_name_matches = 0
    exact_address_matches = 0
    name_country_matches = 0
    address_country_matches = 0

    name_candidates = 0
    address_candidates = 0

    recovered_name = 0
    recovered_address = 0
    recovered_name_country = 0
    recovered_address_country = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk["business_name"] = (
            chunk["business_name"].fillna("")
        )

        chunk["business_address"] = (
            chunk["business_address"].fillna("")
        )

        chunk["name_norm"] = (
            chunk["business_name"]
            .map(normalize_text)
        )

        chunk["address_norm"] = (
            chunk["business_address"]
            .map(normalize_text)
        )

        for _, row in chunk.iterrows():

            total_rows += 1

            entity_id = row["entity_id"]

            name = row["name_norm"]
            address = row["address_norm"]
            country = row["country"]

            true_s1_entities = set()

            # -------------------------------------------------
            # Find S1 candidates by exact normalized name
            # -------------------------------------------------

            name_s1_indices = []

            if name:
                name_s1_indices = name_index.get(
                    name,
                    []
                )

            if name_s1_indices:

                exact_name_matches += 1

                name_candidates += len(
                    name_s1_indices
                )

                for s1_idx in name_s1_indices:

                    s1_id = s1.iloc[
                        s1_idx
                    ]["entity_id"]

                    if entity_id in ground_truth.get(
                        s1_id,
                        set()
                    ):
                        true_s1_entities.add(s1_id)

                if true_s1_entities:
                    recovered_name += 1

            # -------------------------------------------------
            # Exact normalized address
            # -------------------------------------------------

            address_s1_indices = []

            if address:
                address_s1_indices = address_index.get(
                    address,
                    []
                )

            if address_s1_indices:

                exact_address_matches += 1

                address_candidates += len(
                    address_s1_indices
                )

                for s1_idx in address_s1_indices:

                    s1_id = s1.iloc[
                        s1_idx
                    ]["entity_id"]

                    if entity_id in ground_truth.get(
                        s1_id,
                        set()
                    ):
                        true_s1_entities.add(s1_id)

                if any(
                    entity_id in ground_truth.get(
                        s1.iloc[idx]["entity_id"],
                        set()
                    )
                    for idx in address_s1_indices
                ):
                    recovered_address += 1

            # -------------------------------------------------
            # Name + country
            # -------------------------------------------------

            if name:

                nc_indices = name_country_index.get(
                    (name, country),
                    []
                )

                if nc_indices:

                    name_country_matches += 1

                    if any(
                        entity_id in ground_truth.get(
                            s1.iloc[idx]["entity_id"],
                            set()
                        )
                        for idx in nc_indices
                    ):
                        recovered_name_country += 1

            # -------------------------------------------------
            # Address + country
            # -------------------------------------------------

            if address:

                ac_indices = address_country_index.get(
                    (address, country),
                    []
                )

                if ac_indices:

                    address_country_matches += 1

                    if any(
                        entity_id in ground_truth.get(
                            s1.iloc[idx]["entity_id"],
                            set()
                        )
                        for idx in ac_indices
                    ):
                        recovered_address_country += 1

        print(
            f"Processed {total_rows:,} {source_name} records...",
            end="\r"
        )

    print()
    print(f"Total records: {total_rows:,}")

    print("\nBlocking statistics:")

    print(
        f"Exact normalized name matches: "
        f"{exact_name_matches:,} "
        f"({exact_name_matches / total_rows * 100:.2f}%)"
    )

    print(
        f"Exact normalized address matches: "
        f"{exact_address_matches:,} "
        f"({exact_address_matches / total_rows * 100:.2f}%)"
    )

    print(
        f"Name + country matches: "
        f"{name_country_matches:,} "
        f"({name_country_matches / total_rows * 100:.2f}%)"
    )

    print(
        f"Address + country matches: "
        f"{address_country_matches:,} "
        f"({address_country_matches / total_rows * 100:.2f}%)"
    )

    print("\nTrue-match recovery:")

    print(
        f"Recovered through name: "
        f"{recovered_name:,}"
    )

    print(
        f"Recovered through address: "
        f"{recovered_address:,}"
    )

    print(
        f"Recovered through name + country: "
        f"{recovered_name_country:,}"
    )

    print(
        f"Recovered through address + country: "
        f"{recovered_address_country:,}"
    )


# =========================================================
# RUN ANALYSIS
# =========================================================

analyze_source(
    S2_FILE,
    "Source 2",
    "S2"
)

analyze_source(
    S3_FILE,
    "Source 3",
    "S3"
)


print()
print("=" * 70)
print("BLOCKING ANALYSIS COMPLETE")
print("=" * 70)