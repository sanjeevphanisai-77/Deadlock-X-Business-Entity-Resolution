import os
import re
import math
import heapq
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from difflib import SequenceMatcher
from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

TRAIN_DIR = os.path.join(ROOT, "dataset", "train")
TEST_DIR = os.path.join(ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(ROOT, "submission", "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

S1_TRAIN = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_TRAIN = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_TRAIN = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_TRAIN = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

S1_TEST = os.path.join(TEST_DIR, "test_source1.tsv")
S2_TEST = os.path.join(TEST_DIR, "test_source2.tsv")
S3_TEST = os.path.join(TEST_DIR, "test_source3.tsv")

MATCHING_OUT = os.path.join(
    OUTPUT_DIR, "matching_results.tsv"
)

CANDIDATE_OUT = os.path.join(
    OUTPUT_DIR, "candidate_pairs.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_SAMPLE = 5000

TOP_CANDIDATES = 80

MAX_POSTING = 100000

THRESHOLD = 0.75

RANDOM_STATE = 42


# ============================================================
# NORMALIZATION
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
    "near", "opposite", "opp",
    "plot", "block", "district", "city",
    "town", "state", "country"
}


def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower()

    x = re.sub(
        r"[^a-z0-9\s]",
        " ",
        x
    )

    x = re.sub(
        r"\s+",
        " ",
        x
    ).strip()

    return x


def name_tokens(x):
    t = set(normalize_text(x).split())

    return {
        z for z in t
        if z not in LEGAL_WORDS
        and len(z) >= 2
    }


def address_tokens(x):
    t = set(normalize_text(x).split())

    return {
        z for z in t
        if z not in ADDRESS_STOP
        and len(z) >= 2
    }


def number_tokens(x):

    if pd.isna(x):
        return set()

    return set(
        re.findall(
            r"\d+",
            str(x)
        )
    )


# ============================================================
# RECORD PREPROCESSING
# ============================================================

def preprocess(df):

    records = []

    for row in df.to_dict("records"):

        records.append({
            "entity_id": row["entity_id"],

            "name":
                normalize_text(
                    row["business_name"]
                ),

            "name_tokens":
                name_tokens(
                    row["business_name"]
                ),

            "address":
                normalize_text(
                    row["business_address"]
                ),

            "addr_tokens":
                address_tokens(
                    row["business_address"]
                ),

            "numbers":
                number_tokens(
                    row["business_address"]
                ),

            "country":
                str(row["country"]).lower()
                if not pd.isna(row["country"])
                else ""
        })

    return records


# ============================================================
# INDEX
# ============================================================

def build_index(records):

    name_index = defaultdict(list)
    addr_index = defaultdict(list)

    name_freq = Counter()
    addr_freq = Counter()

    for i, r in enumerate(records):

        for token in r["name_tokens"]:
            name_index[token].append(i)
            name_freq[token] += 1

        for token in r["addr_tokens"]:
            addr_index[token].append(i)
            addr_freq[token] += 1

    return (
        name_index,
        addr_index,
        name_freq,
        addr_freq
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    source_record,
    name_index,
    addr_index,
    name_freq,
    addr_freq
):

    scores = defaultdict(float)

    # ----------------------------
    # Name blocking
    # ----------------------------

    for token in source_record["name_tokens"]:

        posting = name_index.get(token)

        if not posting:
            continue

        if len(posting) > MAX_POSTING:
            continue

        # Rare tokens receive larger weight
        weight = 1.0 / math.sqrt(
            max(1, name_freq[token])
        )

        for idx in posting:
            scores[idx] += 3.0 * weight


    # ----------------------------
    # Address blocking
    # ----------------------------

    for token in source_record["addr_tokens"]:

        posting = addr_index.get(token)

        if not posting:
            continue

        if len(posting) > MAX_POSTING:
            continue

        weight = 1.0 / math.sqrt(
            max(1, addr_freq[token])
        )

        for idx in posting:
            scores[idx] += 2.0 * weight


    if not scores:
        return []


    return heapq.nlargest(
        TOP_CANDIDATES,
        scores.keys(),
        key=lambda x: scores[x]
    )


# ============================================================
# FEATURES
# ============================================================

def make_features(a, b):

    nt1 = a["name_tokens"]
    nt2 = b["name_tokens"]

    at1 = a["addr_tokens"]
    at2 = b["addr_tokens"]

    name_union = nt1 | nt2
    addr_union = at1 | at2

    name_inter = nt1 & nt2
    addr_inter = at1 & at2

    name_jaccard = (
        len(name_inter) /
        len(name_union)
        if name_union else 0
    )

    addr_jaccard = (
        len(addr_inter) /
        len(addr_union)
        if addr_union else 0
    )

    name_overlap = (
        len(name_inter) /
        len(nt1)
        if nt1 else 0
    )

    addr_overlap = (
        len(addr_inter) /
        len(at1)
        if at1 else 0
    )

    name_seq = (
        SequenceMatcher(
            None,
            a["name"],
            b["name"]
        ).ratio()
        if a["name"] and b["name"]
        else 0
    )

    addr_seq = (
        SequenceMatcher(
            None,
            a["address"],
            b["address"]
        ).ratio()
        if a["address"] and b["address"]
        else 0
    )

    country_match = int(
        a["country"] != ""
        and a["country"] == b["country"]
    )

    number_match = int(
        bool(
            a["numbers"] &
            b["numbers"]
        )
    )

    return [
        name_jaccard,
        name_overlap,
        name_seq,

        addr_jaccard,
        addr_overlap,
        addr_seq,

        country_match,
        number_match,

        len(nt1),
        len(nt2),

        len(at1),
        len(at2),

        len(name_inter),
        len(addr_inter)
    ]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("FINAL PREDICTION PIPELINE")
print("=" * 70)

print("\n[1/7] Loading training data...")

train_s1 = pd.read_csv(
    S1_TRAIN,
    sep="\t"
)

train_s2 = pd.read_csv(
    S2_TRAIN,
    sep="\t"
)

train_s3 = pd.read_csv(
    S3_TRAIN,
    sep="\t"
)

gt = pd.read_csv(
    GT_TRAIN,
    sep="\t"
)

print(
    f"S1: {len(train_s1):,} | "
    f"S2: {len(train_s2):,} | "
    f"S3: {len(train_s3):,}"
)


# ============================================================
# PREPROCESS TRAINING SOURCES
# ============================================================

print("\n[2/7] Preprocessing training sources...")

tr_s2 = preprocess(train_s2)
tr_s3 = preprocess(train_s3)

print("Training preprocessing complete.")


# ============================================================
# BUILD TRAINING INDEXES
# ============================================================

print("\n[3/7] Building training indexes...")

s2_name_idx, s2_addr_idx, s2_nf, s2_af = \
    build_index(tr_s2)

s3_name_idx, s3_addr_idx, s3_nf, s3_af = \
    build_index(tr_s3)

s2_lookup = {
    r["entity_id"]: r
    for r in tr_s2
}

s3_lookup = {
    r["entity_id"]: r
    for r in tr_s3
}

print(
    f"S2 name tokens: {len(s2_name_idx):,}"
)

print(
    f"S3 name tokens: {len(s3_name_idx):,}"
)


# ============================================================
# TRAIN MATCHER
# ============================================================

print("\n[4/7] Training matcher...")

rng = np.random.default_rng(
    RANDOM_STATE
)

sample_ids = rng.choice(
    train_s1["entity_id"].values,
    size=min(
        TRAIN_SAMPLE,
        len(train_s1)
    ),
    replace=False
)

s1_lookup = {
    row["entity_id"]: row
    for row in preprocess(
        train_s1[
            train_s1["entity_id"].isin(sample_ids)
        ]
    )
}

gt_lookup = {}

for row in gt.to_dict("records"):

    ids = row["matched_entity_ids"]

    if pd.isna(ids) or str(ids).strip() == "":
        gt_lookup[
            row["source1_entity_id"]
        ] = set()

    else:
        gt_lookup[
            row["source1_entity_id"]
        ] = set(
            x.strip()
            for x in str(ids).split(",")
            if x.strip()
        )


X = []
Y = []

for count, (eid, a) in enumerate(
    s1_lookup.items(),
    1
):

    true_ids = gt_lookup.get(
        eid,
        set()
    )

    # S2 candidates
    c2 = generate_candidates(
        a,
        s2_name_idx,
        s2_addr_idx,
        s2_nf,
        s2_af
    )

    # S3 candidates
    c3 = generate_candidates(
        a,
        s3_name_idx,
        s3_addr_idx,
        s3_nf,
        s3_af
    )

    candidate_ids = []

    for idx in c2:
        candidate_ids.append(
            tr_s2[idx]["entity_id"]
        )

    for idx in c3:
        candidate_ids.append(
            tr_s3[idx]["entity_id"]
        )

    # Always include true training matches
    candidate_ids.extend(
        list(true_ids)
    )

    candidate_ids = list(
        dict.fromkeys(candidate_ids)
    )

    positives = []
    negatives = []

    for cid in candidate_ids:

        if cid in true_ids:
            positives.append(cid)
        else:
            negatives.append(cid)

    # Positive pairs
    for cid in positives:

        if cid.startswith("S2-"):
            b = s2_lookup.get(cid)
        else:
            b = s3_lookup.get(cid)

        if b:
            X.append(
                make_features(a, b)
            )
            Y.append(1)

    # Maximum 5 negatives per S1
    for cid in negatives[:5]:

        if cid.startswith("S2-"):
            b = s2_lookup.get(cid)
        else:
            b = s3_lookup.get(cid)

        if b:
            X.append(
                make_features(a, b)
            )
            Y.append(0)

    if count % 500 == 0:
        print(
            f"Training progress: "
            f"{count:,}/{len(s1_lookup):,}"
        )


X = np.asarray(
    X,
    dtype=np.float32
)

Y = np.asarray(Y)

print(
    f"Training pairs: {len(Y):,}"
)

model = HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_leaf_nodes=31,
    random_state=RANDOM_STATE
)

model.fit(X, Y)

import joblib

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "trained_matcher.joblib"
)

joblib.dump(model, MODEL_PATH)

print(f"Matcher trained and saved: {MODEL_PATH}", flush=True)


# ============================================================
# LOAD TEST S1
# ============================================================

print("\n[5/7] Loading test data...")

test_s1 = pd.read_csv(
    S1_TEST,
    sep="\t"
)

test_s2 = pd.read_csv(
    S2_TEST,
    sep="\t"
)

test_s3 = pd.read_csv(
    S3_TEST,
    sep="\t"
)

print(
    f"Test S1: {len(test_s1):,}"
)

print(
    f"Test S2: {len(test_s2):,}"
)

print(
    f"Test S3: {len(test_s3):,}"
)


# ============================================================
# PREPROCESS TEST DATA
# ============================================================

print("\n[6/7] Preprocessing test data...")

te_s1 = preprocess(test_s1)
te_s2 = preprocess(test_s2)
te_s3 = preprocess(test_s3)

print("Test preprocessing complete.")


# ============================================================
# TEST INDEXES
# ============================================================

print("\nBuilding test indexes...")

te_s2_name, te_s2_addr, te_s2_nf, te_s2_af = \
    build_index(te_s2)

te_s3_name, te_s3_addr, te_s3_nf, te_s3_af = \
    build_index(te_s3)


# ============================================================
# FINAL INFERENCE
# ============================================================

print("\n[7/7] Generating final predictions...")

matching_rows = []
candidate_rows = []

for count, a in enumerate(
    te_s1,
    1
):

    # ----------------------------
    # Generate S2 candidates
    # ----------------------------

    c2 = generate_candidates(
        a,
        te_s2_name,
        te_s2_addr,
        te_s2_nf,
        te_s2_af
    )

    # ----------------------------
    # Generate S3 candidates
    # ----------------------------

    c3 = generate_candidates(
        a,
        te_s3_name,
        te_s3_addr,
        te_s3_nf,
        te_s3_af
    )

    all_candidates = []

    for idx in c2:
        all_candidates.append(
            te_s2[idx]
        )

    for idx in c3:
        all_candidates.append(
            te_s3[idx]
        )

    # Remove duplicates
    seen = set()
    unique_candidates = []

    for b in all_candidates:

        if b["entity_id"] not in seen:
            seen.add(
                b["entity_id"]
            )
            unique_candidates.append(b)

    candidate_ids = [
        b["entity_id"]
        for b in unique_candidates
    ]

    # ----------------------------
    # Score candidates
    # ----------------------------

    predicted = []

    if unique_candidates:

        features = np.asarray(
            [
                make_features(a, b)
                for b in unique_candidates
            ],
            dtype=np.float32
        )

        probabilities = model.predict_proba(
            features
        )[:, 1]

        for b, probability in zip(
            unique_candidates,
            probabilities
        ):

            if probability >= THRESHOLD:

                predicted.append(
                    b["entity_id"]
                )

    # ----------------------------
    # Candidate output
    # ----------------------------

    candidate_rows.append({
        "source1_entity_id":
            a["entity_id"],

        "candidate_entity_ids":
            ",".join(candidate_ids)
    })

    # ----------------------------
    # Matching output
    # ----------------------------

    matching_rows.append({
        "source1_entity_id":
            a["entity_id"],

        "matched_entity_ids":
            ",".join(
                dict.fromkeys(predicted)
            )
    })

    if count % 10000 == 0:

        print(
            f"Inference progress: "
            f"{count:,}/{len(te_s1):,}"
        )


# ============================================================
# WRITE OUTPUT
# ============================================================

print("\nWriting output files...")

pd.DataFrame(
    matching_rows,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
).to_csv(
    MATCHING_OUT,
    sep="\t",
    index=False
)

pd.DataFrame(
    candidate_rows,
    columns=[
        "source1_entity_id",
        "candidate_entity_ids"
    ]
).to_csv(
    CANDIDATE_OUT,
    sep="\t",
    index=False
)


print("\n" + "=" * 70)
print("FINAL PREDICTION COMPLETE")
print("=" * 70)

print(
    f"Matching file: {MATCHING_OUT}"
)

print(
    f"Candidate file: {CANDIDATE_OUT}"
)

print(
    f"S1 entities processed: {len(te_s1):,}"
)

print("\nNext step: run the official validator.")