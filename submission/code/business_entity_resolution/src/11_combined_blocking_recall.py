import pandas as pd
import re
import unicodedata

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

    replacements = {
        r"\broad\b": "rd",
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bdrive\b": "dr",
        r"\blane\b": "ln",
        r"\bhighway\b": "hwy",
        r"\bparkway\b": "pkwy",
        r"\bplace\b": "pl",
        r"\bcourt\b": "ct",
        r"\bapartment\b": "apt",
        r"\bfloor\b": "fl",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Name tokens
# ---------------------------------------------------------

def normalized_name_tokens(text):

    text = normalize_text(text)

    tokens = text.split()

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

    return {
        token
        for token in tokens
        if token not in suffixes
    }


# ---------------------------------------------------------
# Address tokens
# ---------------------------------------------------------

def normalized_address_tokens(text):

    text = normalize_text(text)

    tokens = text.split()

    generic_tokens = {
        "road",
        "rd",
        "street",
        "st",
        "avenue",
        "ave",
        "drive",
        "dr",
        "lane",
        "ln",
        "place",
        "pl",
        "city",
        "county",
        "district",
        "state",
        "floor",
        "fl",
        "unit",
        "no",
        "number",
    }

    return {
        token
        for token in tokens
        if token not in generic_tokens
        and len(token) >= 2
    }


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
# Fast S1 lookup
# ---------------------------------------------------------

print("\nCreating S1 lookup...")

s1_lookup = (
    s1.set_index("entity_id")
    .to_dict("index")
)

print("S1 lookup created.")


# ---------------------------------------------------------
# Load validation ground truth
# ---------------------------------------------------------

print("\nLoading validation ground truth...")

validation_gt = pd.read_csv(
    f"{TRAIN_DIR}/splits/validation_ground_truth.tsv",
    sep="\t",
    dtype=str
)

print(
    f"Validation entities: "
    f"{len(validation_gt):,}"
)


# ---------------------------------------------------------
# Collect validation S2 IDs
# ---------------------------------------------------------

print("\nCollecting validation S2 IDs...")

validation_s2_ids = set()

for value in validation_gt["matched_entity_ids"].dropna():

    value = str(value).strip()

    if not value:
        continue

    for entity_id in value.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            validation_s2_ids.add(entity_id)


print(
    f"Unique validation S2 IDs: "
    f"{len(validation_s2_ids):,}"
)


# ---------------------------------------------------------
# Load required S2 records
# ---------------------------------------------------------

print("\nReading Source 2 in chunks...")

s2_lookup = {}

for chunk in pd.read_csv(
    f"{TRAIN_DIR}/train_source2.tsv",
    sep="\t",
    dtype=str,
    usecols=[
        "entity_id",
        "business_name",
        "business_address"
    ],
    chunksize=250_000
):

    mask = chunk["entity_id"].isin(
        validation_s2_ids
    )

    selected = chunk.loc[mask]

    for row in selected.itertuples(index=False):

        s2_lookup[row.entity_id] = (
            row.business_name,
            row.business_address
        )

    print(
        f"Collected S2 records: "
        f"{len(s2_lookup):,}"
    )


print(
    f"\nS2 validation records loaded: "
    f"{len(s2_lookup):,}"
)


# ---------------------------------------------------------
# Evaluate combined blocking
# ---------------------------------------------------------

print("\nEvaluating combined blocking...")
print("=" * 70)

total_true_matches = 0

name_recovered = 0
address_recovered = 0
combined_recovered = 0

entities_zero = 0
entities_partial = 0
entities_all = 0

processed = 0


for gt_row in validation_gt.itertuples(index=False):

    s1_id = gt_row.source1_entity_id
    matched_ids = gt_row.matched_entity_ids

    if pd.isna(matched_ids):
        continue

    matched_ids = str(matched_ids).strip()

    if not matched_ids:
        continue

    true_ids = [
        x.strip()
        for x in matched_ids.split(",")
        if x.strip().startswith("S2-")
    ]

    if not true_ids:
        continue

    s1_record = s1_lookup.get(s1_id)

    if s1_record is None:
        continue

    s1_name_tokens = normalized_name_tokens(
        s1_record["business_name"]
    )

    s1_address_tokens = normalized_address_tokens(
        s1_record["business_address"]
    )

    entity_combined_recovered = 0

    for match_id in true_ids:

        candidate = s2_lookup.get(match_id)

        if candidate is None:
            continue

        s2_name, s2_address = candidate

        s2_name_tokens = normalized_name_tokens(
            s2_name
        )

        s2_address_tokens = normalized_address_tokens(
            s2_address
        )

        name_match = bool(
            s1_name_tokens &
            s2_name_tokens
        )

        address_match = bool(
            s1_address_tokens &
            s2_address_tokens
        )

        if name_match:
            name_recovered += 1

        if address_match:
            address_recovered += 1

        # Combined candidate rule:
        # Name overlap OR address overlap
        if name_match or address_match:
            combined_recovered += 1
            entity_combined_recovered += 1

    total_true_matches += len(true_ids)

    if entity_combined_recovered == 0:

        entities_zero += 1

    elif entity_combined_recovered < len(true_ids):

        entities_partial += 1

    else:

        entities_all += 1

    processed += 1

    if processed % 10_000 == 0:

        print(
            f"Processed: {processed:,}"
        )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("COMBINED BLOCKING RESULTS")
print("=" * 70)

print(
    f"Validation entities with S2 matches: "
    f"{processed:,}"
)

print(
    f"Total true S2 matches: "
    f"{total_true_matches:,}"
)

print("\nName-token blocking:")
print(
    f"Recovered: {name_recovered:,}"
)

if total_true_matches:
    print(
        f"Recall: "
        f"{name_recovered / total_true_matches * 100:.4f}%"
    )


print("\nAddress-token blocking:")
print(
    f"Recovered: {address_recovered:,}"
)

if total_true_matches:
    print(
        f"Recall: "
        f"{address_recovered / total_true_matches * 100:.4f}%"
    )


print("\nName OR Address blocking:")
print(
    f"Recovered: {combined_recovered:,}"
)

if total_true_matches:
    combined_recall = (
        combined_recovered /
        total_true_matches
    ) * 100

    print(
        f"Recall: "
        f"{combined_recall:.4f}%"
    )


print(
    f"\nEntities with ZERO recall: "
    f"{entities_zero:,}"
)

print(
    f"Entities with PARTIAL recall: "
    f"{entities_partial:,}"
)

print(
    f"Entities with ALL matches recovered: "
    f"{entities_all:,}"
)

print("\nCombined blocking experiment completed.")