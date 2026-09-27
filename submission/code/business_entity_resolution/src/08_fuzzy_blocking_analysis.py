import pandas as pd
import re
import unicodedata
from collections import defaultdict
from difflib import SequenceMatcher

TRAIN_DIR = "dataset/train"

# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove accents / diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower()

    # Common address abbreviations
    replacements = {
        r"\broad\b": "rd",
        r"\broad\.": "rd",
        r"\bstreet\b": "st",
        r"\bst\.": "st",
        r"\bavenue\b": "ave",
        r"\bave\.": "ave",
        r"\bdrive\b": "dr",
        r"\bdr\.": "dr",
        r"\blane\b": "ln",
        r"\bln\.": "ln",
        r"\broadway\b": "rdway",
        r"\bhighway\b": "hwy",
        r"\bparkway\b": "pkwy",
        r"\bplace\b": "pl",
        r"\bpl\.": "pl",
        r"\bcourt\b": "ct",
        r"\bapartment\b": "apt",
        r"\bapt\.": "apt",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Punctuation → spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove duplicate spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def normalized_name_tokens(text):
    text = normalize_text(text)

    tokens = text.split()

    # Remove very common legal/business suffixes
    suffixes = {
        "llc",
        "ltd",
        "limited",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "llp",
        "plc",
    }

    tokens = [t for t in tokens if t not in suffixes]

    return set(tokens)

def name_tokens(text):
    return set(normalize_text(text).split())


def jaccard_similarity(a, b):
    a = name_tokens(a)
    b = name_tokens(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def fuzzy_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------
# Load Source 1
# ---------------------------------------------------------

print("Loading Source 1...")

s1 = pd.read_csv(
    f"{TRAIN_DIR}/train_source1.tsv",
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


# ---------------------------------------------------------
# Build lightweight blocking indexes
# ---------------------------------------------------------

print("\nBuilding blocking indexes...")

name_index = defaultdict(list)
address_index = defaultdict(list)

for row in s1.itertuples(index=False):

    name = normalize_text(row.business_name)
    address = normalize_text(row.business_address)

    if name:
        name_index[name].append(row.entity_id)

    if address:
        address_index[address].append(row.entity_id)

print(f"Unique normalized names: {len(name_index):,}")
print(f"Unique normalized addresses: {len(address_index):,}")


# ---------------------------------------------------------
# Test fuzzy similarity on missed-match examples
# ---------------------------------------------------------

print("\nTesting fuzzy similarity...")
print("=" * 70)

examples = [
    ("Crystal Staffing Solutions LLC",
     "Crystal Solutions LLC Partners"),

    ("Fayth Fritz Lifeco Corp",
     "Fayth Fet Lifeco Corp"),

    ("Jeanlouis Advantage LLC",
     "Jeanl0uis Advantage LLC"),

    ("Schubert Adr LLC",
     "Schubert LLC Adr"),

    ("Vadodara Industries Pvt Ltd",
     "Vadodara Industries Pvt Limited"),

    ("Orellana Investments LLC",
     "Orellana Investments Investments Llc"),

    ("Uptown Pub",
     "UPTOWN PUB CO"),

    ("Modern Soulpower LLC",
     "Modern LLC Center"),
]


for name1, name2 in examples:

    fuzzy = fuzzy_similarity(name1, name2)
    jac = jaccard_similarity(name1, name2)

    print("\nS1 :", name1)
    print("S2 :", name2)
    print(f"Fuzzy    : {fuzzy:.4f}")
    print(f"Jaccard  : {jac:.4f}")


# ---------------------------------------------------------
# Analyze fuzzy thresholds
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("Fuzzy threshold experiment")
print("=" * 70)

thresholds = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]

for threshold in thresholds:

    passed = 0

    for row in examples:

        score = fuzzy_similarity(row[0], row[1])

        if score >= threshold:
            passed += 1

    print(
        f"Threshold {threshold:.2f} -> "
        f"{passed}/{len(examples)} examples pass"
    )


print("\nFuzzy blocking analysis completed.")