import pandas as pd
import re
from collections import Counter

S2_FILE = "dataset/train/train_source2.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)

    return " ".join(text.split())


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
    "sa", "spa"
}


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
    "floor", "fl",
    "building", "bldg",
    "unit", "suite", "ste"
}


def name_tokens(text):
    tokens = normalize_text(text).split()

    return {
        t for t in tokens
        if len(t) >= 3 and t not in LEGAL_SUFFIXES
    }


def address_tokens(text):
    tokens = normalize_text(text).split()

    return {
        t for t in tokens
        if len(t) >= 3 and t not in ADDRESS_GENERIC
    }


# ============================================================
# COUNT TOKEN FREQUENCIES
# ============================================================

name_freq = Counter()
address_freq = Counter()

print("Scanning Source 2...")

for chunk in pd.read_csv(
    S2_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "business_name",
        "business_address"
    ],
    chunksize=200_000
):

    for name in chunk["business_name"].fillna(""):
        name_freq.update(name_tokens(name))

    for address in chunk["business_address"].fillna(""):
        address_freq.update(address_tokens(address))


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MOST COMMON NAME TOKENS")
print("=" * 70)

for token, count in name_freq.most_common(50):
    print(f"{token:30s} {count:>10,}")


print("\n" + "=" * 70)
print("MOST COMMON ADDRESS TOKENS")
print("=" * 70)

for token, count in address_freq.most_common(50):
    print(f"{token:30s} {count:>10,}")


# ============================================================
# FREQUENCY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TOKEN FREQUENCY DISTRIBUTION")
print("=" * 70)

thresholds = [
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
    25000,
    50000,
    100000
]

print("\nNAME TOKENS")

for threshold in thresholds:
    count = sum(
        1 for freq in name_freq.values()
        if freq >= threshold
    )

    print(
        f">= {threshold:>6,}: "
        f"{count:>8,} tokens"
    )


print("\nADDRESS TOKENS")

for threshold in thresholds:
    count = sum(
        1 for freq in address_freq.values()
        if freq >= threshold
    )

    print(
        f">= {threshold:>6,}: "
        f"{count:>8,} tokens"
    )


print("=" * 70)