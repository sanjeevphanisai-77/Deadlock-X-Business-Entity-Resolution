import pandas as pd
import re
import numpy as np
from collections import defaultdict, Counter

BASE = "dataset"

S1_FILE = f"{BASE}/train/train_source1.tsv"
S2_FILE = f"{BASE}/train/train_source2.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = text.replace("&", " and ")

    # Keep letters and numbers
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())


# ============================================================
# NAME TOKENS
# ============================================================

LEGAL_SUFFIXES = {
    "inc", "incorporated",
    "llc", "ltd", "limited",
    "corp", "corporation",
    "co", "company",
    "plc",
    "pvt", "private",
    "llp",
    "gmbh",
    "ag",
    "sa",
    "spa"
}


def name_tokens(text):
    text = normalize_text(text)

    tokens = text.split()

    return {
        t for t in tokens
        if len(t) >= 3 and t not in LEGAL_SUFFIXES
    }


# ============================================================
# ADDRESS TOKENS
# ============================================================

ADDRESS_GENERIC = {
    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "lane", "ln",
    "drive", "dr",
    "boulevard", "blvd",
    "highway", "hwy",
    "way",
    "parkway", "pkwy",
    "place", "pl",
    "court", "ct",
    "circle", "cir",
    "floor",
    "fl",
    "building",
    "bldg",
    "unit",
    "suite",
    "ste"
}


def address_tokens(text):
    text = normalize_text(text)

    tokens = text.split()

    return {
        t for t in tokens
        if len(t) >= 3 and t not in ADDRESS_GENERIC
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

print(f"S1 records: {len(s1):,}")


# ============================================================
# BUILD TOKEN FREQUENCIES FROM S2
# ============================================================

print("\nScanning Source 2 for token frequencies...")

name_freq = Counter()
address_freq = Counter()

chunk_size = 200_000

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    chunksize=chunk_size
):

    for name in chunk["business_name"].fillna(""):
        name_freq.update(name_tokens(name))

    for address in chunk["business_address"].fillna(""):
        address_freq.update(address_tokens(address))


print(f"Unique name tokens: {len(name_freq):,}")
print(f"Unique address tokens: {len(address_freq):,}")


# ============================================================
# INFORMATIVE TOKENS
# ============================================================

# Ignore extremely common tokens.
#
# These thresholds are deliberately conservative.
#
# Name token appearing in > 5,000 records
# Address token appearing in > 20,000 records

MAX_NAME_FREQ = 5_000
MAX_ADDRESS_FREQ = 20_000


useful_name_tokens = {
    token
    for token, freq in name_freq.items()
    if freq <= MAX_NAME_FREQ
}

useful_address_tokens = {
    token
    for token, freq in address_freq.items()
    if freq <= MAX_ADDRESS_FREQ
}


print("\nInformative tokens:")
print(f"Name tokens:    {len(useful_name_tokens):,}")
print(f"Address tokens: {len(useful_address_tokens):,}")


# ============================================================
# BUILD COMPACT INDEXES
# ============================================================

print("\nBuilding compact S2 indexes...")

name_index = defaultdict(set)
address_index = defaultdict(set)

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address"
    ],
    chunksize=chunk_size
):

    for row in chunk.itertuples(index=False):

        entity_id = row.entity_id

        for token in name_tokens(row.business_name):

            if token in useful_name_tokens:
                name_index[token].add(entity_id)

        for token in address_tokens(row.business_address):

            if token in useful_address_tokens:
                address_index[token].add(entity_id)


print(f"Name index entries:    {len(name_index):,}")
print(f"Address index entries: {len(address_index):,}")


# ============================================================
# CANDIDATE SIZE SAMPLING
# ============================================================

print("\nCalculating candidate sizes using a sample...")
print("=" * 70)

# Instead of processing all 2.2M S1 records,
# first analyse 50,000 representative records.

SAMPLE_SIZE = 50_000

sample = s1.sample(
    n=min(SAMPLE_SIZE, len(s1)),
    random_state=42
)


candidate_sizes = []

zero_candidates = 0

for i, row in enumerate(sample.itertuples(index=False), start=1):

    candidates = set()

    # Name blocking
    for token in name_tokens(row.business_name):

        if token in useful_name_tokens:
            candidates.update(
                name_index.get(token, ())
            )

    # Address blocking
    for token in address_tokens(row.business_address):

        if token in useful_address_tokens:
            candidates.update(
                address_index.get(token, ())
            )

    size = len(candidates)

    candidate_sizes.append(size)

    if size == 0:
        zero_candidates += 1

    if i % 5000 == 0:
        print(f"Processed {i:,}/{len(sample):,}")


# ============================================================
# STATISTICS
# ============================================================

sizes = np.array(candidate_sizes)

print("\n" + "=" * 70)
print("CANDIDATE SIZE RESULTS")
print("=" * 70)

print(f"Sample size:       {len(sizes):,}")
print(f"Mean:              {sizes.mean():,.2f}")
print(f"Median:            {np.median(sizes):,.0f}")
print(f"75th percentile:   {np.percentile(sizes, 75):,.0f}")
print(f"90th percentile:   {np.percentile(sizes, 90):,.0f}")
print(f"95th percentile:   {np.percentile(sizes, 95):,.0f}")
print(f"99th percentile:   {np.percentile(sizes, 99):,.0f}")
print(f"99.9th percentile: {np.percentile(sizes, 99.9):,.0f}")
print(f"Maximum:           {sizes.max():,}")
print(f"Zero candidates:   {zero_candidates:,}")

print("\nCandidate-size distribution:")

bins = [
    0,
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    float("inf")
]

labels = [
    "0",
    "1-5",
    "6-10",
    "11-25",
    "26-50",
    "51-100",
    "101-250",
    "251-500",
    "501-1000",
    "1001-2500",
    "2501-5000",
    "5001-10000",
    "10000+"
]

counts = np.histogram(
    sizes,
    bins=bins
)[0]

for label, count in zip(labels, counts):

    percentage = count / len(sizes) * 100

    print(
        f"{label:>12}: "
        f"{count:>7,} "
        f"({percentage:6.2f}%)"
    )

print("=" * 70)
print("Analysis complete.")