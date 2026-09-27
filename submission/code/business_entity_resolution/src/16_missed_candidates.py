import pandas as pd
import re
from collections import defaultdict

BASE = "dataset"

S1_FILE = f"{BASE}/train/train_source1.tsv"
S2_FILE = f"{BASE}/train/train_source2.tsv"
GT_FILE = f"{BASE}/train/splits/validation_ground_truth.tsv"

SAMPLE_SIZE = 5000


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "inc", "incorporated",
    "llc", "ltd", "limited",
    "corp", "corporation",
    "co", "company",
    "plc", "pvt", "private",
    "llp", "gmbh", "ag", "sa", "spa"
}

ADDRESS_GENERIC = {
    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "lane", "ln",
    "drive", "dr",
    "boulevard", "blvd",
    "highway", "hwy",
    "way", "parkway", "pkwy",
    "place", "pl",
    "court", "ct",
    "circle", "cir",
    "floor", "fl",
    "building", "bldg",
    "unit", "suite", "ste"
}


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def name_tokens(value):
    return {
        x
        for x in normalize_text(value).split()
        if len(x) >= 3 and x not in LEGAL_SUFFIXES
    }


def address_tokens(value):
    return {
        x
        for x in normalize_text(value).split()
        if len(x) >= 3 and x not in ADDRESS_GENERIC
    }


def address_numbers(value):
    return {
        x
        for x in normalize_text(value).split()
        if any(c.isdigit() for c in x)
    }


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

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


# ============================================================
# LOAD VALIDATION GROUND TRUTH
# ============================================================

print("Loading validation ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")


# ============================================================
# SAME 5,000 VALIDATION SAMPLE
# ============================================================

gt_with_matches = gt[
    gt["matched_entity_ids"].str.len() > 0
]

sample_gt = gt_with_matches.sample(
    n=min(SAMPLE_SIZE, len(gt_with_matches)),
    random_state=42
)

sample_ids = set(
    sample_gt["source1_entity_id"]
)

sample_s1 = s1[
    s1["entity_id"].isin(sample_ids)
].copy()


# ============================================================
# TRUE MATCHES
# ============================================================

true_matches = {}

needed_s2_ids = set()

for row in sample_gt.itertuples(index=False):

    matches = {
        x.strip()
        for x in row.matched_entity_ids.split(",")
        if x.strip()
    }

    true_matches[row.source1_entity_id] = matches

    for x in matches:

        if x.startswith("S2"):
            needed_s2_ids.add(x)


print(
    f"Validation sample: {len(sample_s1):,}"
)

print(
    f"True S2 records needed: "
    f"{len(needed_s2_ids):,}"
)


# ============================================================
# LOAD REQUIRED S2 RECORDS
# ============================================================

print("\nScanning Source 2...")

needed_s2_chunks = []

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    chunksize=250_000
):

    found = chunk[
        chunk["entity_id"].isin(needed_s2_ids)
    ]

    if not found.empty:
        needed_s2_chunks.append(found)


s2 = pd.concat(
    needed_s2_chunks,
    ignore_index=True
)

print(
    f"Loaded S2 records: {len(s2):,}"
)


# ============================================================
# BUILD INDEXES
# ============================================================

print("\nBuilding indexes...")

name_index = defaultdict(set)
address_index = defaultdict(set)
number_index = defaultdict(set)

for row in s2.itertuples(index=False):

    sid = row.entity_id

    for token in name_tokens(row.business_name):
        name_index[token].add(sid)

    for token in address_tokens(row.business_address):
        address_index[token].add(sid)

    for number in address_numbers(row.business_address):
        number_index[number].add(sid)


# ============================================================
# FIND MISSED TRUE MATCHES
# ============================================================

missed = []

print("\nSearching for missed matches...")

for row in sample_s1.itertuples(index=False):

    s1_id = row.entity_id

    true_s2 = {
        x
        for x in true_matches[s1_id]
        if x.startswith("S2")
    }

    if not true_s2:
        continue

    candidates = set()

    # Name blocking
    for token in name_tokens(row.business_name):
        candidates.update(
            name_index.get(token, ())
        )

    # Address blocking
    for token in address_tokens(row.business_address):
        candidates.update(
            address_index.get(token, ())
        )

    # Address number blocking
    for number in address_numbers(row.business_address):
        candidates.update(
            number_index.get(number, ())
        )

    missed_ids = true_s2 - candidates

    for s2_id in missed_ids:

        s2_row = s2[
            s2["entity_id"] == s2_id
        ]

        if s2_row.empty:
            continue

        s2_row = s2_row.iloc[0]

        missed.append({
            "s1_id": s1_id,
            "s1_name": row.business_name,
            "s1_address": row.business_address,
            "s1_country": row.country,

            "s2_id": s2_id,
            "s2_name": s2_row["business_name"],
            "s2_address": s2_row["business_address"],
            "s2_country": s2_row["country"]
        })


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MISSED TRUE MATCHES")
print("=" * 80)

print(
    f"Total missed true matches: {len(missed)}"
)

for i, item in enumerate(missed, start=1):

    print("\n" + "-" * 80)

    print(f"MISSED MATCH #{i}")

    print(f"\nS1 ID:      {item['s1_id']}")
    print(f"S1 Name:    {item['s1_name']}")
    print(f"S1 Address: {item['s1_address']}")
    print(f"S1 Country: {item['s1_country']}")

    print(f"\nS2 ID:      {item['s2_id']}")
    print(f"S2 Name:    {item['s2_name']}")
    print(f"S2 Address: {item['s2_address']}")
    print(f"S2 Country: {item['s2_country']}")

print("\n" + "=" * 80)