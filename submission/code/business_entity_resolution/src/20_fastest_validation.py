import os
import re
import sys
import math
import heapq
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from difflib import SequenceMatcher
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
S1_FILE = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_FILE = os.path.join(TRAIN_DIR, "train_source2.tsv")
GT_FILE = os.path.join(TRAIN_DIR, "splits", "train_ground_truth.tsv")
VAL_FILE = os.path.join(TRAIN_DIR, "splits", "validation_ground_truth.tsv")


# ============================================================
# SETTINGS
# ============================================================

N_TRAIN_ENTITIES = 3000
N_VALIDATION_ENTITIES = 500

TOP_K = 100

RANDOM_STATE = 42


# ============================================================
# TEXT NORMALIZATION
# ============================================================

LEGAL_WORDS = {
    "llc", "inc", "incorporated", "ltd", "limited",
    "corp", "corporation", "co", "company",
    "pvt", "private", "plc", "lp", "llp"
}

ADDRESS_STOP = {
    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "drive", "dr", "building", "bldg",
    "floor", "fl", "suite", "ste", "unit",
    "near", "opposite", "opp", "plot", "block",
    "district", "city", "town", "state", "country"
}


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def tokens(x, remove_legal=False):
    x = normalize_text(x)
    t = set(x.split())

    if remove_legal:
        t -= LEGAL_WORDS

    return t


def address_tokens(x):
    t = tokens(x)

    # Keep meaningful alphanumeric tokens
    t = {
        z for z in t
        if len(z) >= 2 and z not in ADDRESS_STOP
    }

    return t


def number_tokens(x):
    if pd.isna(x):
        return set()

    return set(re.findall(r"\d+", str(x)))


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FAST VALIDATION")
print("=" * 70)

print("\nLoading Source 1...")
s1 = pd.read_csv(S1_FILE, sep="\t")
print(f"S1 rows: {len(s1):,}")

print("\nLoading Source 2...")
s2 = pd.read_csv(S2_FILE, sep="\t")
print(f"S2 rows: {len(s2):,}")

print("\nLoading validation ground truth...")
val_gt = pd.read_csv(VAL_FILE, sep="\t")
print(f"Validation rows: {len(val_gt):,}")


# ============================================================
# SAMPLE VALIDATION ENTITIES
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

available = val_gt["source1_entity_id"].values

selected_ids = rng.choice(
    available,
    size=min(N_VALIDATION_ENTITIES, len(available)),
    replace=False
)

selected_ids = set(selected_ids)

val_gt_small = val_gt[
    val_gt["source1_entity_id"].isin(selected_ids)
].copy()

s1_small = s1[
    s1["entity_id"].isin(selected_ids)
].copy()

print(
    f"\nValidation sample: {len(s1_small):,} entities"
)


# ============================================================
# PREPROCESS SOURCE 2 ONCE
# ============================================================

print("\nPreprocessing Source 2...")

s2["name_tokens"] = s2["business_name"].map(
    lambda x: tokens(x, remove_legal=True)
)

s2["addr_tokens"] = s2["business_address"].map(
    address_tokens
)

s2["number_tokens"] = s2["business_address"].map(
    number_tokens
)

s2["country_norm"] = (
    s2["country"]
    .fillna("")
    .astype(str)
    .str.lower()
)

s2["name_norm"] = s2["business_name"].map(normalize_text)
s2["addr_norm"] = s2["business_address"].map(normalize_text)

print("Source 2 preprocessing complete.")


# ============================================================
# BUILD INVERTED INDEXES
# ============================================================

print("\nBuilding indexes...")

name_index = defaultdict(list)
addr_index = defaultdict(list)

for idx, row in enumerate(
    s2[
        ["name_tokens", "addr_tokens"]
    ].itertuples(index=False)
):

    nt, at = row

    for t in nt:
        name_index[t].append(idx)

    for t in at:
        addr_index[t].append(idx)

print(f"Name index tokens: {len(name_index):,}")
print(f"Address index tokens: {len(addr_index):,}")


# ============================================================
# PRECOMPUTE SOURCE 1
# ============================================================

print("\nPreprocessing Source 1...")

s1_records = {}

for row in s1_small.to_dict("records"):

    eid = row["entity_id"]

    s1_records[eid] = {
        "name": normalize_text(row["business_name"]),
        "name_tokens": tokens(
            row["business_name"],
            remove_legal=True
        ),
        "addr": normalize_text(row["business_address"]),
        "addr_tokens": address_tokens(
            row["business_address"]
        ),
        "numbers": number_tokens(
            row["business_address"]
        ),
        "country": str(
            row["country"]
        ).lower()
        if not pd.isna(row["country"])
        else ""
    }


# ============================================================
# GROUND TRUTH
# ============================================================

truth = {}

for row in val_gt_small.to_dict("records"):

    ids = row["matched_entity_ids"]

    if pd.isna(ids) or str(ids).strip() == "":
        truth[row["source1_entity_id"]] = set()

    else:
        truth[row["source1_entity_id"]] = set(
            x.strip()
            for x in str(ids).split(",")
            if x.strip()
        )


# ============================================================
# FAST CANDIDATE GENERATION
# ============================================================

def generate_candidates(s1r):

    scores = Counter()

    # Name tokens
    for token in s1r["name_tokens"]:

        posting = name_index.get(token)

        if posting is None:
            continue

        # Ignore extremely common tokens
        if len(posting) > 100000:
            continue

        for idx in posting:
            scores[idx] += 3


    # Address tokens
    for token in s1r["addr_tokens"]:

        posting = addr_index.get(token)

        if posting is None:
            continue

        if len(posting) > 100000:
            continue

        for idx in posting:
            scores[idx] += 2


    # Country boost
    if not scores:
        return []

    # Only cheap integer ranking here
    best = heapq.nlargest(
        TOP_K,
        scores.items(),
        key=lambda x: x[1]
    )

    return [
        idx for idx, score in best
    ]


# ============================================================
# EXPENSIVE FEATURES ONLY FOR TOP K
# ============================================================

def make_features(s1r, s2r):

    s1n = s1r["name"]
    s2n = s2r["name_norm"]

    s1a = s1r["addr"]
    s2a = s2r["addr_norm"]

    nt1 = s1r["name_tokens"]
    nt2 = s2r["name_tokens"]

    at1 = s1r["addr_tokens"]
    at2 = s2r["addr_tokens"]

    num1 = s1r["numbers"]
    num2 = s2r["number_tokens"]

    name_union = nt1 | nt2
    addr_union = at1 | at2

    name_jaccard = (
        len(nt1 & nt2) / len(name_union)
        if name_union else 0
    )

    addr_jaccard = (
        len(at1 & at2) / len(addr_union)
        if addr_union else 0
    )

    name_overlap = (
        len(nt1 & nt2) / len(nt1)
        if nt1 else 0
    )

    addr_overlap = (
        len(at1 & at2) / len(at1)
        if at1 else 0
    )

    name_seq = (
        SequenceMatcher(None, s1n, s2n).ratio()
        if s1n and s2n else 0
    )

    addr_seq = (
        SequenceMatcher(None, s1a, s2a).ratio()
        if s1a and s2a else 0
    )

    country_match = int(
        s1r["country"] != ""
        and s1r["country"] == s2r["country_norm"]
    )

    number_overlap = int(
        len(num1 & num2) > 0
    )

    return [
        name_jaccard,
        name_overlap,
        name_seq,
        addr_jaccard,
        addr_overlap,
        addr_seq,
        country_match,
        number_overlap,
        len(nt1),
        len(nt2),
        len(at1),
        len(at2)
    ]


# ============================================================
# CREATE TRAINING PAIRS
# ============================================================

print("\nCreating training pairs...")

train_ids = rng.choice(
    available,
    size=min(N_TRAIN_ENTITIES, len(available)),
    replace=False
)

train_ids = set(train_ids)

train_gt = val_gt[
    val_gt["source1_entity_id"].isin(train_ids)
]

train_s1 = s1[
    s1["entity_id"].isin(train_ids)
]

train_lookup = {}

for row in train_s1.to_dict("records"):

    train_lookup[row["entity_id"]] = {
        "name": normalize_text(row["business_name"]),
        "name_tokens": tokens(
            row["business_name"],
            remove_legal=True
        ),
        "addr": normalize_text(row["business_address"]),
        "addr_tokens": address_tokens(
            row["business_address"]
        ),
        "numbers": number_tokens(
            row["business_address"]
        ),
        "country": str(row["country"]).lower()
        if not pd.isna(row["country"])
        else ""
    }


X_train = []
y_train = []

for counter_i, row in enumerate(
    train_gt.to_dict("records"),
    1
):

    eid = row["source1_entity_id"]

    if eid not in train_lookup:
        continue

    s1r = train_lookup[eid]

    ids = row["matched_entity_ids"]

    positives = set()

    if not pd.isna(ids):
        positives = set(
            x.strip()
            for x in str(ids).split(",")
            if x.strip()
        )

    candidates = generate_candidates(s1r)

    # Ensure true matches are present
    true_indexes = []

    for pid in positives:

        matches = s2.index[
            s2["entity_id"] == pid
        ].tolist()

        if matches:
            true_indexes.append(matches[0])

    candidate_indexes = set(candidates)

    candidate_indexes.update(true_indexes)

    # positives
    for idx in true_indexes:

        X_train.append(
            make_features(
                s1r,
                s2.iloc[idx]
            )
        )

        y_train.append(1)

    # negatives
    negatives = [
        idx for idx in candidate_indexes
        if idx not in set(true_indexes)
    ]

    # maximum 5 negatives/entity
    for idx in negatives[:5]:

        X_train.append(
            make_features(
                s1r,
                s2.iloc[idx]
            )
        )

        y_train.append(0)

    if counter_i % 250 == 0:
        print(
            f"Training entities processed: "
            f"{counter_i:,}/{len(train_gt):,}"
        )


print(f"\nTraining pairs: {len(X_train):,}")


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining HistGradientBoosting model...")

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train)

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

print("Model trained.")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

all_results = []

for i, (eid, s1r) in enumerate(
    s1_records.items(),
    1
):

    candidate_indexes = generate_candidates(s1r)

    rows = []

    for idx in candidate_indexes:

        rows.append(
            make_features(
                s1r,
                s2.iloc[idx]
            )
        )

    if rows:

        probs = model.predict_proba(
            np.asarray(rows, dtype=np.float32)
        )[:, 1]

        ids = [
            s2.iloc[idx]["entity_id"]
            for idx in candidate_indexes
        ]

    else:

        probs = []
        ids = []

    all_results.append(
        {
            "source1": eid,
            "ids": ids,
            "probs": probs
        }
    )

    if i % 25 == 0:
        print(
            f"Validation progress: "
            f"{i:,}/{len(s1_records):,}"
        )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)

thresholds = [
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

for threshold in thresholds:

    entity_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for result in all_results:

        eid = result["source1"]

        actual = truth.get(eid, set())

        predicted = {
            eid2
            for eid2, p in zip(
                result["ids"],
                result["probs"]
            )
            if p >= threshold
        }

        tp = len(predicted & actual)
        fp = len(predicted - actual)
        fn = len(actual - predicted)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if not predicted and not actual:
            f05 = 1.0

        else:

            precision = (
                tp / len(predicted)
                if predicted else 0
            )

            recall = (
                tp / len(actual)
                if actual else 0
            )

            if precision == 0 and recall == 0:
                f05 = 0

            else:
                f05 = (
                    1.25 * precision * recall
                    /
                    (0.25 * precision + recall)
                )

        entity_scores.append(f05)

    macro_f05 = np.mean(entity_scores)

    precision = (
        total_tp /
        (total_tp + total_fp)
        if total_tp + total_fp
        else 0
    )

    recall = (
        total_tp /
        (total_tp + total_fn)
        if total_tp + total_fn
        else 0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"Macro F0.5={macro_f05:.4f}"
    )


print("\nDONE.")