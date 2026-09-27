import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
GT_PATH = "dataset/train/splits/train_ground_truth.tsv"

SAMPLE_SIZE = 5000
MAX_NEGATIVES_PER_POSITIVE = 3
RANDOM_STATE = 42


# ============================================================
# TEXT NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "llp",
    "pvt", "private", "plc", "gmbh", "sa", "ag"
}


def normalize_unicode(text):
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)

    return text


def normalize_ascii(text):
    text = normalize_unicode(text)

    text = unicodedata.normalize(
        "NFKD", text
    ).encode(
        "ascii", "ignore"
    ).decode("ascii")

    return text


def normalize_text(text):
    text = normalize_ascii(text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text, remove_legal=False):
    text = normalize_text(text)

    tokens = text.split()

    if remove_legal:
        tokens = [
            token for token in tokens
            if token not in LEGAL_SUFFIXES
        ]

    return tokens


def token_set(text, remove_legal=False):
    return set(tokenize(text, remove_legal))


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


def overlap_similarity(a, b):
    if not a or not b:
        return 0.0

    intersection = len(a & b)

    return intersection / min(len(a), len(b))


def sequence_similarity(a, b):
    from difflib import SequenceMatcher

    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def number_tokens(text):
    text = normalize_text(text)

    return set(
        re.findall(r"\b\d+\b", text)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def make_features(s1, s2):
    """
    s1 and s2 are dictionaries.
    """

    s1_name = s1.get("business_name", "")
    s2_name = s2.get("business_name", "")

    s1_address = s1.get("business_address", "")
    s2_address = s2.get("business_address", "")

    s1_country = normalize_text(
        s1.get("country", "")
    )

    s2_country = normalize_text(
        s2.get("country", "")
    )

    # --------------------------------------------------------
    # NAME FEATURES
    # --------------------------------------------------------

    name_unicode_1 = normalize_unicode(s1_name)
    name_unicode_2 = normalize_unicode(s2_name)

    name_ascii_1 = normalize_text(s1_name)
    name_ascii_2 = normalize_text(s2_name)

    name_tokens_1 = token_set(
        s1_name,
        remove_legal=True
    )

    name_tokens_2 = token_set(
        s2_name,
        remove_legal=True
    )

    # --------------------------------------------------------
    # ADDRESS FEATURES
    # --------------------------------------------------------

    address_unicode_1 = normalize_unicode(s1_address)
    address_unicode_2 = normalize_unicode(s2_address)

    address_ascii_1 = normalize_text(s1_address)
    address_ascii_2 = normalize_text(s2_address)

    address_tokens_1 = token_set(s1_address)
    address_tokens_2 = token_set(s2_address)

    # --------------------------------------------------------
    # NUMBER FEATURES
    # --------------------------------------------------------

    numbers_1 = number_tokens(s1_address)
    numbers_2 = number_tokens(s2_address)

    if numbers_1 or numbers_2:
        number_overlap = len(
            numbers_1 & numbers_2
        ) / max(
            1,
            min(
                len(numbers_1),
                len(numbers_2)
            )
        )
    else:
        number_overlap = 0.0

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = {
        # Name
        "name_exact_unicode":
            int(
                name_unicode_1 != ""
                and name_unicode_1 == name_unicode_2
            ),

        "name_exact_ascii":
            int(
                name_ascii_1 != ""
                and name_ascii_1 == name_ascii_2
            ),

        "name_jaccard":
            jaccard_similarity(
                name_tokens_1,
                name_tokens_2
            ),

        "name_overlap":
            overlap_similarity(
                name_tokens_1,
                name_tokens_2
            ),

        "name_char_similarity":
            sequence_similarity(
                s1_name,
                s2_name
            ),

        "name_length_difference":
            abs(
                len(name_ascii_1)
                -
                len(name_ascii_2)
            ),

        # Address
        "address_exact_unicode":
            int(
                address_unicode_1 != ""
                and address_unicode_1 == address_unicode_2
            ),

        "address_exact_ascii":
            int(
                address_ascii_1 != ""
                and address_ascii_1 == address_ascii_2
            ),

        "address_jaccard":
            jaccard_similarity(
                address_tokens_1,
                address_tokens_2
            ),

        "address_overlap":
            overlap_similarity(
                address_tokens_1,
                address_tokens_2
            ),

        "address_char_similarity":
            sequence_similarity(
                s1_address,
                s2_address
            ),

        "address_number_overlap":
            number_overlap,

        "address_length_difference":
            abs(
                len(address_ascii_1)
                -
                len(address_ascii_2)
            ),

        # Country
        "country_match":
            int(
                s1_country != ""
                and s1_country == s2_country
            )
    }

    return features


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S1 records: {len(s1):,}")


print("\nLoading Source 2...")

s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print(f"S2 records: {len(s2):,}")


print("\nLoading training ground truth...")

gt = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str,
    keep_default_na=False
)


# ============================================================
# SAMPLE SOURCE 1
# ============================================================

print(f"\nTraining sample: {SAMPLE_SIZE:,}")

sample_s1 = (
    s1
    .sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)


# ============================================================
# GROUND TRUTH LOOKUP
# ============================================================

gt_lookup = {}

for row in gt.to_dict("records"):

    s1_id = row["source1_entity_id"]

    matched_ids = row.get(
        "matched_entity_ids",
        ""
    )

    if pd.isna(matched_ids):
        matched_ids = ""

    matched_ids = str(
        matched_ids
    ).strip()

    if matched_ids:
        ids = {
            x.strip()
            for x in matched_ids.split(",")
            if x.strip()
        }
    else:
        ids = set()

    gt_lookup[s1_id] = ids


# ============================================================
# BUILD S2 LOOKUP
# ============================================================

print("\nBuilding S2 lookup...")

s2_records = s2.to_dict("records")

s2_lookup = {
    row["entity_id"]: row
    for row in s2_records
}


# ============================================================
# BUILD BLOCKING INDEXES
# ============================================================

print("Building blocking indexes...")

name_index = {}
address_index = {}


for row in s2_records:

    entity_id = row["entity_id"]

    # -------------------------------
    # Name index
    # -------------------------------

    name_tokens = token_set(
        row.get("business_name", ""),
        remove_legal=True
    )

    for token in name_tokens:

        if len(token) < 2:
            continue

        name_index.setdefault(
            token,
            set()
        ).add(entity_id)

    # -------------------------------
    # Address index
    # -------------------------------

    address_tokens = token_set(
        row.get("business_address", "")
    )

    for token in address_tokens:

        if len(token) < 2:
            continue

        address_index.setdefault(
            token,
            set()
        ).add(entity_id)


# ============================================================
# CREATE TRAINING PAIRS
# ============================================================

print("\nCreating training pairs...")

X_rows = []
y_rows = []

positive_count = 0
negative_count = 0

sample_records = sample_s1.to_dict("records")


for counter, s1_row in enumerate(
    sample_records,
    start=1
):

    s1_id = s1_row["entity_id"]

    true_ids = gt_lookup.get(
        s1_id,
        set()
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = set()

    # Name blocking
    name_tokens = token_set(
        s1_row.get("business_name", ""),
        remove_legal=True
    )

    for token in name_tokens:

        if len(token) < 2:
            continue

        candidates.update(
            name_index.get(
                token,
                set()
            )
        )

    # Address blocking
    address_tokens = token_set(
        s1_row.get("business_address", "")
    )

    for token in address_tokens:

        if len(token) < 2:
            continue

        candidates.update(
            address_index.get(
                token,
                set()
            )
        )

    # --------------------------------------------------------
    # Positive pairs
    # --------------------------------------------------------

    for s2_id in true_ids:

        s2_row = s2_lookup.get(s2_id)

        if s2_row is None:
            continue

        features = make_features(
            s1_row,
            s2_row
        )

        X_rows.append(features)
        y_rows.append(1)

        positive_count += 1

    # --------------------------------------------------------
    # Negative pairs
    # --------------------------------------------------------

    negative_candidates = [
        candidate
        for candidate in candidates
        if candidate not in true_ids
    ]

    # Deterministic sampling
    negative_candidates = sorted(
        negative_candidates
    )[:MAX_NEGATIVES_PER_POSITIVE * max(
        1,
        len(true_ids)
    )]

    for s2_id in negative_candidates:

        s2_row = s2_lookup.get(s2_id)

        if s2_row is None:
            continue

        features = make_features(
            s1_row,
            s2_row
        )

        X_rows.append(features)
        y_rows.append(0)

        negative_count += 1

    if counter % 500 == 0:

        print(
            f"Processed {counter:,}/{SAMPLE_SIZE:,} "
            f"| positives={positive_count:,} "
            f"| negatives={negative_count:,}"
        )


# ============================================================
# BUILD TRAINING DATAFRAME
# ============================================================

print("\nBuilding feature dataframe...")

X = pd.DataFrame(X_rows)
y = np.array(y_rows)


print(f"Training pairs: {len(X):,}")
print(f"Positive pairs: {positive_count:,}")
print(f"Negative pairs: {negative_count:,}")

print("\nFeature columns:")

for column in X.columns:
    print("  ", column)


# ============================================================
# HANDLE NUMERIC TYPES
# ============================================================

X = X.astype(float)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining HistGradientBoostingClassifier...")

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    random_state=RANDOM_STATE
)


model.fit(
    X,
    y
)


print("Model training complete.")


# ============================================================
# TRAINING PREDICTIONS
# ============================================================

print("\nGenerating training probabilities...")

probabilities = model.predict_proba(
    X
)[:, 1]


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\nThreshold evaluation:")

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95
]


best_threshold = None
best_f05 = -1


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    if precision == 0 and recall == 0:
        f05 = 0.0
    else:
        f05 = (
            1.25
            * precision
            * recall
        ) / (
            0.25 * precision
            + recall
        )

    print(
        f"Threshold={threshold:.2f} "
        f"| Precision={precision:.4f} "
        f"| Recall={recall:.4f} "
        f"| F0.5={f05:.4f}"
    )

    if f05 > best_f05:

        best_f05 = f05
        best_threshold = threshold


# ============================================================
# FINAL PROTOTYPE RESULT
# ============================================================

print("\n" + "=" * 60)

print("PROTOTYPE TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best training threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Best training F0.5: "
    f"{best_f05:.4f}"
)

print(
    f"Training pairs: "
    f"{len(X):,}"
)

print(
    f"Positive pairs: "
    f"{positive_count:,}"
)

print(
    f"Negative pairs: "
    f"{negative_count:,}"
)

print("=" * 60)

print(
    "\nIMPORTANT: This is only a prototype "
    "training evaluation."
)

print(
    "The final threshold must be selected "
    "using the held-out validation set "
    "and the challenge's per-Source-1 "
    "macro F0.5 metric."
)