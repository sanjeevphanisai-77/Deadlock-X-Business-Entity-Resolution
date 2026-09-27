import pandas as pd
import re
from collections import defaultdict


# ============================================================
# FILES
# ============================================================

BASE = "dataset"

S1_FILE = f"{BASE}/train/train_source1.tsv"
S2_FILE = f"{BASE}/train/train_source2.tsv"
GT_FILE = f"{BASE}/train/splits/validation_ground_truth.tsv"


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 5000

# We only keep S2 records that are actually relevant to the
# validation sample. This makes experimentation fast.


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
    "place", "pl", "court", "ct",
    "circle", "cir", "floor", "fl",
    "building", "bldg", "unit",
    "suite", "ste"
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
# LOAD S1
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
# SELECT 5,000 VALIDATION ENTITIES
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

print(
    f"Validation sample: {len(sample_s1):,}"
)


# ============================================================
# GET TRUE S2 IDS
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
    f"True S2 records needed: "
    f"{len(needed_s2_ids):,}"
)


# ============================================================
# LOAD ONLY RELEVANT S2 RECORDS
# ============================================================

print("\nScanning Source 2...")

needed_s2 = []

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
        needed_s2.append(found)

s2_true = pd.concat(
    needed_s2,
    ignore_index=True
)

print(
    f"Loaded true S2 records: "
    f"{len(s2_true):,}"
)


# ============================================================
# BUILD A SMALL BLOCKING INDEX
# ============================================================

print("\nBuilding validation index...")

name_index = defaultdict(set)
address_index = defaultdict(set)
number_index = defaultdict(set)

for row in s2_true.itertuples(index=False):

    sid = row.entity_id

    for token in name_tokens(row.business_name):
        name_index[token].add(sid)

    for token in address_tokens(row.business_address):
        address_index[token].add(sid)

    for number in address_numbers(row.business_address):
        number_index[number].add(sid)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(row):

    candidates = set()

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    for token in name_tokens(row.business_name):
        candidates.update(
            name_index.get(token, ())
        )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    for token in address_tokens(row.business_address):
        candidates.update(
            address_index.get(token, ())
        )

    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    for number in address_numbers(row.business_address):
        candidates.update(
            number_index.get(number, ())
        )

    return candidates


# ============================================================
# EVALUATION
# ============================================================

total_true = 0
total_recovered = 0
total_candidates = 0

zero = 0
partial = 0
full = 0

candidate_sizes = []

print("\nEvaluating candidates...")
print("=" * 70)

for i, row in enumerate(
    sample_s1.itertuples(index=False),
    start=1
):

    true_ids = {
        x for x in true_matches[row.entity_id]
        if x.startswith("S2")
    }

    candidates = generate_candidates(row)

    hits = candidates.intersection(true_ids)

    total_true += len(true_ids)
    total_recovered += len(hits)
    total_candidates += len(candidates)

    candidate_sizes.append(
        len(candidates)
    )

    if len(hits) == 0:
        zero += 1

    elif len(hits) == len(true_ids):
        full += 1

    else:
        partial += 1

    if i % 500 == 0:
        print(
            f"Processed {i:,}/{len(sample_s1):,}"
        )


# ============================================================
# RESULTS
# ============================================================

candidate_sizes.sort()

recall = (
    total_recovered / total_true
    if total_true
    else 0
)

candidate_precision = (
    total_recovered / total_candidates
    if total_candidates
    else 0
)

print("\n" + "=" * 70)
print("FAST VALIDATION RESULTS")
print("=" * 70)

print(
    f"Validation entities: {len(sample_s1):,}"
)

print(
    f"True S2 matches:      {total_true:,}"
)

print(
    f"Recovered matches:    {total_recovered:,}"
)

print(
    f"Candidate recall:     {recall * 100:.4f}%"
)

print(
    f"Candidate precision:  {candidate_precision * 100:.6f}%"
)

print(
    f"Total candidates:     {total_candidates:,}"
)

print(
    f"Average candidates:   "
    f"{total_candidates / len(sample_s1):,.2f}"
)

print(
    f"Median candidates:    "
    f"{candidate_sizes[len(candidate_sizes)//2]:,}"
)

print(
    f"90th percentile:      "
    f"{candidate_sizes[int(len(candidate_sizes)*0.90)]:,}"
)

print(
    f"95th percentile:      "
    f"{candidate_sizes[int(len(candidate_sizes)*0.95)]:,}"
)

print(
    f"99th percentile:      "
    f"{candidate_sizes[int(len(candidate_sizes)*0.99)]:,}"
)

print(
    f"Maximum candidates:   "
    f"{max(candidate_sizes):,}"
)

print(
    f"Zero recall:          {zero:,}"
)

print(
    f"Partial recall:       {partial:,}"
)

print(
    f"Full recall:          {full:,}"
)

print("=" * 70)