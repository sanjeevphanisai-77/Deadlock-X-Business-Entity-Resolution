import pandas as pd
import re
from collections import Counter, defaultdict

BASE = "dataset"

S1_FILE = f"{BASE}/train/train_source1.tsv"
S2_FILE = f"{BASE}/train/train_source2.tsv"
GT_FILE = f"{BASE}/train/splits/validation_ground_truth.tsv"


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
    "ag", "sa", "spa"
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
# LOAD VALIDATION DATA
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


print("\nLoading validation ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
)

gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")


# ============================================================
# BUILD VALIDATION GROUND TRUTH
# ============================================================

true_matches = {}

validation_s1_ids = set()

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id

    matches = set()

    if row.matched_entity_ids:

        matches = set(
            x.strip()
            for x in row.matched_entity_ids.split(",")
            if x.strip()
        )

    true_matches[s1_id] = matches

    if matches:
        validation_s1_ids.add(s1_id)


print(
    f"Validation S1 entities with matches: "
    f"{len(validation_s1_ids):,}"
)


# ============================================================
# COLLECT TRUE S2 IDS
# ============================================================

true_s2_ids = set()

for matches in true_matches.values():
    true_s2_ids.update(
        x for x in matches
        if x.startswith("S2")
    )


print(
    f"Unique true S2 IDs: "
    f"{len(true_s2_ids):,}"
)


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

print("\nScanning S2 token frequencies...")

name_freq = Counter()
address_freq = Counter()

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


print(
    f"Name vocabulary: "
    f"{len(name_freq):,}"
)

print(
    f"Address vocabulary: "
    f"{len(address_freq):,}"
)


# ============================================================
# BUILD MULTIPLE BLOCKING INDEXES
# ============================================================

configs = [
    ("1000 / 2500", 1000, 2500),
    ("500 / 1000", 500, 1000),
    ("250 / 500", 250, 500),
    ("100 / 250", 100, 250),
]


results = []


for config_name, max_name_freq, max_address_freq in configs:

    print("\n" + "=" * 70)
    print(
        f"TESTING: {config_name}"
    )
    print("=" * 70)

    useful_name = {
        token
        for token, freq in name_freq.items()
        if freq <= max_name_freq
    }

    useful_address = {
        token
        for token, freq in address_freq.items()
        if freq <= max_address_freq
    }

    print(
        f"Useful name tokens: "
        f"{len(useful_name):,}"
    )

    print(
        f"Useful address tokens: "
        f"{len(useful_address):,}"
    )


    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    name_index = defaultdict(set)
    address_index = defaultdict(set)

    print("Building indexes...")

    for chunk in pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
        usecols=[
            "entity_id",
            "business_name",
            "business_address"
        ],
        chunksize=200_000
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            for token in name_tokens(row.business_name):

                if token in useful_name:
                    name_index[token].add(entity_id)

            for token in address_tokens(row.business_address):

                if token in useful_address:
                    address_index[token].add(entity_id)


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    recovered = 0
    total_true_matches = 0

    candidate_total = 0
    candidate_sizes = []

    zero_recall = 0
    partial_recall = 0
    full_recall = 0

    print("Evaluating validation entities...")

    for i, s1_id in enumerate(validation_s1_ids, start=1):

        row = s1[
            s1["entity_id"] == s1_id
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        true_ids = true_matches[s1_id]

        true_s2 = {
            x for x in true_ids
            if x.startswith("S2")
        }

        if not true_s2:
            continue

        candidates = set()

        # Name blocking
        for token in name_tokens(row["business_name"]):

            if token in useful_name:
                candidates.update(
                    name_index.get(token, ())
                )

        # Address blocking
        for token in address_tokens(row["business_address"]):

            if token in useful_address:
                candidates.update(
                    address_index.get(token, ())
                )

        candidate_sizes.append(len(candidates))
        candidate_total += len(candidates)

        hit = len(
            candidates.intersection(true_s2)
        )

        recovered += hit
        total_true_matches += len(true_s2)

        if hit == 0:
            zero_recall += 1
        elif hit == len(true_s2):
            full_recall += 1
        else:
            partial_recall += 1

        if i % 25_000 == 0:
            print(
                f"Processed {i:,}/"
                f"{len(validation_s1_ids):,}"
            )


    recall = (
        recovered / total_true_matches
        if total_true_matches
        else 0
    )

    average_candidates = (
        candidate_total / len(candidate_sizes)
        if candidate_sizes
        else 0
    )

    candidate_sizes.sort()

    def percentile(p):
        if not candidate_sizes:
            return 0

        index = int(
            p * (len(candidate_sizes) - 1)
        )

        return candidate_sizes[index]


    print("\nRESULT")

    print(
        f"Recall:             "
        f"{recall * 100:.6f}%"
    )

    print(
        f"Recovered:          "
        f"{recovered:,}"
    )

    print(
        f"True matches:       "
        f"{total_true_matches:,}"
    )

    print(
        f"Average candidates: "
        f"{average_candidates:,.2f}"
    )

    print(
        f"Median candidates:  "
        f"{percentile(0.50):,}"
    )

    print(
        f"90th percentile:    "
        f"{percentile(0.90):,}"
    )

    print(
        f"95th percentile:    "
        f"{percentile(0.95):,}"
    )

    print(
        f"99th percentile:    "
        f"{percentile(0.99):,}"
    )

    print(
        f"Maximum:            "
        f"{max(candidate_sizes):,}"
    )

    print(
        f"Zero recall:        "
        f"{zero_recall:,}"
    )

    print(
        f"Partial recall:     "
        f"{partial_recall:,}"
    )

    print(
        f"Full recall:        "
        f"{full_recall:,}"
    )

    results.append(
        (
            config_name,
            recall,
            average_candidates,
            percentile(0.50),
            percentile(0.90),
            percentile(0.95),
            percentile(0.99),
            max(candidate_sizes),
            zero_recall
        )
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 100)
print("FINAL COMPARISON")
print("=" * 100)

print(
    f"{'Config':<15}"
    f"{'Recall':>12}"
    f"{'Avg':>12}"
    f"{'Median':>12}"
    f"{'P90':>12}"
    f"{'P95':>12}"
    f"{'P99':>12}"
    f"{'Max':>12}"
)

print("-" * 100)

for r in results:

    print(
        f"{r[0]:<15}"
        f"{r[1] * 100:>11.5f}%"
        f"{r[2]:>12,.0f}"
        f"{r[3]:>12,}"
        f"{r[4]:>12,}"
        f"{r[5]:>12,}"
        f"{r[6]:>12,}"
        f"{r[7]:>12,}"
    )

print("=" * 100)