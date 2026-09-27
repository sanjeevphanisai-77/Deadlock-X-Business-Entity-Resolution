import os
import re
import gc
import math
import heapq
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from difflib import SequenceMatcher
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


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

MODEL_PATH = os.path.join(
    OUTPUT_DIR, "trained_matcher.joblib"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_SAMPLE = 5000
TRAIN_NEGATIVES = 5

TOP_CANDIDATES = 80

# Ignore huge postings.
MAX_POSTING = 100000

# Final probability threshold
THRESHOLD = 0.75

RANDOM_STATE = 42

# Extra fast pre-filter.
# A candidate survives if it has enough shared information.
MIN_SHARED_SCORE = 2.0


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_WORDS = {
    "llc", "inc", "incorporated", "ltd",
    "limited", "corp", "corporation",
    "co", "company", "pvt", "private",
    "plc", "lp", "llp"
}

ADDRESS_STOP = {
    "road", "rd", "street", "st",
    "avenue", "ave", "lane", "ln",
    "drive", "dr", "building", "bldg",
    "floor", "fl", "suite", "ste",
    "unit", "near", "opposite", "opp",
    "plot", "block", "district",
    "city", "town", "state", "country"
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
    text = normalize_text(x)

    return {
        token
        for token in text.split()
        if token not in LEGAL_WORDS
        and len(token) >= 2
    }


def address_tokens(x):
    text = normalize_text(x)

    return {
        token
        for token in text.split()
        if token not in ADDRESS_STOP
        and len(token) >= 2
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
# COMPACT RECORD
# ============================================================

def make_record(row):

    return {
        "entity_id": row["entity_id"],

        "name": normalize_text(
            row["business_name"]
        ),

        "name_tokens": name_tokens(
            row["business_name"]
        ),

        "address": normalize_text(
            row["business_address"]
        ),

        "addr_tokens": address_tokens(
            row["business_address"]
        ),

        "numbers": number_tokens(
            row["business_address"]
        ),

        "country":
            str(row["country"]).lower()
            if not pd.isna(row["country"])
            else ""
    }


# ============================================================
# INDEX
# ============================================================

def build_index(records):

    name_index = defaultdict(list)
    addr_index = defaultdict(list)

    name_freq = Counter()
    addr_freq = Counter()

    for idx, r in enumerate(records):

        for token in r["name_tokens"]:
            name_index[token].append(idx)
            name_freq[token] += 1

        for token in r["addr_tokens"]:
            addr_index[token].append(idx)
            addr_freq[token] += 1

    return (
        name_index,
        addr_index,
        name_freq,
        addr_freq
    )


# ============================================================
# FAST CANDIDATE GENERATOR
# ============================================================

def generate_candidates(
    source,
    name_index,
    addr_index,
    name_freq,
    addr_freq
):

    scores = defaultdict(float)

    # NAME BLOCKING
    for token in source["name_tokens"]:

        posting = name_index.get(token)

        if not posting:
            continue

        if len(posting) > MAX_POSTING:
            continue

        weight = 1.0 / math.sqrt(
            max(
                1,
                name_freq[token]
            )
        )

        for idx in posting:
            scores[idx] += 3.0 * weight

    # ADDRESS BLOCKING
    for token in source["addr_tokens"]:

        posting = addr_index.get(token)

        if not posting:
            continue

        if len(posting) > MAX_POSTING:
            continue

        weight = 1.0 / math.sqrt(
            max(
                1,
                addr_freq[token]
            )
        )

        for idx in posting:
            scores[idx] += 2.0 * weight

    if not scores:
        return []

    # Keep only strongest candidates.
    return heapq.nlargest(
        TOP_CANDIDATES,
        scores,
        key=scores.get
    )


# ============================================================
# FAST FEATURES
# ============================================================

def fast_features(a, b):

    nt1 = a["name_tokens"]
    nt2 = b["name_tokens"]

    at1 = a["addr_tokens"]
    at2 = b["addr_tokens"]

    name_inter = nt1 & nt2
    addr_inter = at1 & at2

    name_union = nt1 | nt2
    addr_union = at1 | at2

    name_jaccard = (
        len(name_inter) / len(name_union)
        if name_union else 0.0
    )

    addr_jaccard = (
        len(addr_inter) / len(addr_union)
        if addr_union else 0.0
    )

    name_overlap = (
        len(name_inter) / len(nt1)
        if nt1 else 0.0
    )

    addr_overlap = (
        len(addr_inter) / len(at1)
        if at1 else 0.0
    )

    # Only calculate expensive SequenceMatcher
    # after the cheap token checks have passed.
    if (
        name_jaccard >= 0.20
        or name_overlap >= 0.25
        or addr_jaccard >= 0.20
        or addr_overlap >= 0.25
    ):
        name_seq = (
            SequenceMatcher(
                None,
                a["name"],
                b["name"]
            ).ratio()
            if a["name"] and b["name"]
            else 0.0
        )

        addr_seq = (
            SequenceMatcher(
                None,
                a["address"],
                b["address"]
            ).ratio()
            if a["address"] and b["address"]
            else 0.0
        )
    else:
        name_seq = 0.0
        addr_seq = 0.0

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
# TRAIN MODEL
# ============================================================

def train_model():

    print("\n" + "=" * 70)
    print("TRAINING MATCHER")
    print("=" * 70)

    print("\nLoading training files...")

    s1 = pd.read_csv(
        S1_TRAIN,
        sep="\t"
    )

    s2 = pd.read_csv(
        S2_TRAIN,
        sep="\t"
    )

    s3 = pd.read_csv(
        S3_TRAIN,
        sep="\t"
    )

    gt = pd.read_csv(
        GT_TRAIN,
        sep="\t"
    )

    print(
        f"S1={len(s1):,} "
        f"S2={len(s2):,} "
        f"S3={len(s3):,}"
    )

    print("\nPreprocessing training S2/S3...")

    s2_records = [
        make_record(row)
        for row in s2.to_dict("records")
    ]

    s3_records = [
        make_record(row)
        for row in s3.to_dict("records")
    ]

    print("Training records prepared.")

    print("\nBuilding indexes...")

    (
        s2_name_idx,
        s2_addr_idx,
        s2_nf,
        s2_af
    ) = build_index(s2_records)

    (
        s3_name_idx,
        s3_addr_idx,
        s3_nf,
        s3_af
    ) = build_index(s3_records)

    print(
        f"S2 name tokens: "
        f"{len(s2_name_idx):,}"
    )

    print(
        f"S3 name tokens: "
        f"{len(s3_name_idx):,}"
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_ids = set(
        rng.choice(
            s1["entity_id"].values,
            size=min(
                TRAIN_SAMPLE,
                len(s1)
            ),
            replace=False
        )
    )

    sample_s1 = s1[
        s1["entity_id"].isin(sample_ids)
    ]

    s1_records = {
        row["entity_id"]:
            make_record(row)
        for row in sample_s1.to_dict("records")
    }

    gt_lookup = {}

    for row in gt.to_dict("records"):

        ids = row["matched_entity_ids"]

        if (
            pd.isna(ids)
            or str(ids).strip() == ""
        ):

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

    s2_lookup = {
        r["entity_id"]: r
        for r in s2_records
    }

    s3_lookup = {
        r["entity_id"]: r
        for r in s3_records
    }

    X = []
    Y = []

    print("\nCreating training pairs...")

    for count, (eid, a) in enumerate(
        s1_records.items(),
        1
    ):

        true_ids = gt_lookup.get(
            eid,
            set()
        )

        c2 = generate_candidates(
            a,
            s2_name_idx,
            s2_addr_idx,
            s2_nf,
            s2_af
        )

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
                s2_records[idx]["entity_id"]
            )

        for idx in c3:
            candidate_ids.append(
                s3_records[idx]["entity_id"]
            )

        # Ensure all positives are present.
        candidate_ids.extend(
            list(true_ids)
        )

        candidate_ids = list(
            dict.fromkeys(candidate_ids)
        )

        positive_ids = [
            x
            for x in candidate_ids
            if x in true_ids
        ]

        negative_ids = [
            x
            for x in candidate_ids
            if x not in true_ids
        ]

        for cid in positive_ids:

            if cid.startswith("S2-"):
                b = s2_lookup.get(cid)
            else:
                b = s3_lookup.get(cid)

            if b:

                X.append(
                    fast_features(a, b)
                )

                Y.append(1)

        for cid in negative_ids[
            :TRAIN_NEGATIVES
        ]:

            if cid.startswith("S2-"):
                b = s2_lookup.get(cid)
            else:
                b = s3_lookup.get(cid)

            if b:

                X.append(
                    fast_features(a, b)
                )

                Y.append(0)

        if count % 500 == 0:
            print(
                f"Training: "
                f"{count:,}/{len(s1_records):,}",
                flush=True
            )

    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(Y)

    print(
        f"\nTraining pairs: {len(Y):,}"
    )

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )

    print("Fitting model...")

    model.fit(
        X,
        Y
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"Model saved: {MODEL_PATH}"
    )

    del s1
    del s2
    del s3
    del gt

    del s2_records
    del s3_records
    del s1_records

    del s2_lookup
    del s3_lookup

    del s2_name_idx
    del s2_addr_idx

    del s3_name_idx
    del s3_addr_idx

    del s2_nf
    del s2_af

    del s3_nf
    del s3_af

    del X
    del Y

    gc.collect()

    print(
        "\nTraining memory released."
    )

    return model


# ============================================================
# TEST LOADER
# ============================================================

def load_test_source(path, label):

    print(
        f"\nLoading {label}..."
    )

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print(
        f"{label} rows: {len(df):,}"
    )

    records = [
        make_record(row)
        for row in df.to_dict("records")
    ]

    del df
    gc.collect()

    return records


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("FINAL FAST PREDICTION")
print("=" * 70)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

model = train_model()

# ------------------------------------------------------------
# LOAD TEST S1
# ------------------------------------------------------------

print(
    "\nLoading TEST Source 1..."
)

test_s1_df = pd.read_csv(
    S1_TEST,
    sep="\t"
)

print(
    f"Test S1: "
    f"{len(test_s1_df):,}"
)

test_s1_records = [
    make_record(row)
    for row in test_s1_df.to_dict(
        "records"
    )
]

del test_s1_df
gc.collect()

# ------------------------------------------------------------
# LOAD S2
# ------------------------------------------------------------

test_s2_records = load_test_source(
    S2_TEST,
    "TEST Source 2"
)

(
    s2_name_idx,
    s2_addr_idx,
    s2_nf,
    s2_af
) = build_index(
    test_s2_records
)

print(
    f"S2 name tokens: "
    f"{len(s2_name_idx):,}"
)

# ------------------------------------------------------------
# LOAD S3
# ------------------------------------------------------------

test_s3_records = load_test_source(
    S3_TEST,
    "TEST Source 3"
)

(
    s3_name_idx,
    s3_addr_idx,
    s3_nf,
    s3_af
) = build_index(
    test_s3_records
)

print(
    f"S3 name tokens: "
    f"{len(s3_name_idx):,}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\nPreparing output files...")

with open(
    MATCHING_OUT,
    "w",
    encoding="utf-8",
    newline=""
) as match_file:

    with open(
        CANDIDATE_OUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as candidate_file:

        match_file.write(
            "source1_entity_id\tmatched_entity_ids\n"
        )

        candidate_file.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        total = len(
            test_s1_records
        )

        print(
            "\nStarting FINAL FAST inference..."
        )

        for count, a in enumerate(
            test_s1_records,
            1
        ):

            # ------------------------------------------------
            # Candidate generation
            # ------------------------------------------------

            c2 = generate_candidates(
                a,
                s2_name_idx,
                s2_addr_idx,
                s2_nf,
                s2_af
            )

            c3 = generate_candidates(
                a,
                s3_name_idx,
                s3_addr_idx,
                s3_nf,
                s3_af
            )

            candidates = []

            for idx in c2:
                candidates.append(
                    test_s2_records[idx]
                )

            for idx in c3:
                candidates.append(
                    test_s3_records[idx]
                )

            # ------------------------------------------------
            # Remove duplicates
            # ------------------------------------------------

            seen = set()
            unique_candidates = []

            for r in candidates:

                eid = r["entity_id"]

                if eid not in seen:

                    seen.add(eid)

                    unique_candidates.append(
                        r
                    )

            # ------------------------------------------------
            # FAST PRE-FILTER
            # ------------------------------------------------

            final_candidates = []

            for b in unique_candidates:

                name_shared = len(
                    a["name_tokens"]
                    &
                    b["name_tokens"]
                )

                addr_shared = len(
                    a["addr_tokens"]
                    &
                    b["addr_tokens"]
                )

                number_match = bool(
                    a["numbers"]
                    &
                    b["numbers"]
                )

                country_match = (
                    a["country"] != ""
                    and
                    a["country"] == b["country"]
                )

                # Keep candidates with useful evidence.
                #
                # Strong name evidence
                # OR strong address evidence
                # OR matching numbers + country
                # OR at least one shared name and address token

                keep = (
                    name_shared >= 2
                    or
                    addr_shared >= 2
                    or
                    (
                        name_shared >= 1
                        and
                        addr_shared >= 1
                    )
                    or
                    (
                        number_match
                        and
                        country_match
                    )
                )

                if keep:
                    final_candidates.append(b)

            # ------------------------------------------------
            # If filtering removed everything, retain the
            # strongest blocking candidate.
            # ------------------------------------------------

            if (
                not final_candidates
                and
                unique_candidates
            ):

                final_candidates = (
                    unique_candidates[:5]
                )

            # ------------------------------------------------
            # Candidate IDs
            # ------------------------------------------------

            candidate_ids = [
                r["entity_id"]
                for r in final_candidates
            ]

            # ------------------------------------------------
            # ML SCORING
            # ------------------------------------------------

            predicted = []

            if final_candidates:

                features = np.asarray(
                    [
                        fast_features(
                            a,
                            b
                        )
                        for b in final_candidates
                    ],
                    dtype=np.float32
                )

                probabilities = (
                    model.predict_proba(
                        features
                    )[:, 1]
                )

                for b, probability in zip(
                    final_candidates,
                    probabilities
                ):

                    if probability >= THRESHOLD:

                        predicted.append(
                            b["entity_id"]
                        )

            # ------------------------------------------------
            # WRITE CANDIDATES
            # ------------------------------------------------

            candidate_file.write(
                a["entity_id"]
                + "\t"
                + ",".join(
                    dict.fromkeys(
                        candidate_ids
                    )
                )
                + "\n"
            )

            # ------------------------------------------------
            # WRITE MATCHES
            # ------------------------------------------------

            match_file.write(
                a["entity_id"]
                + "\t"
                + ",".join(
                    dict.fromkeys(
                        predicted
                    )
                )
                + "\n"
            )

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if count % 5000 == 0:

                print(
                    f"Inference: "
                    f"{count:,}/{total:,} "
                    f"({count / total * 100:.2f}%)",
                    flush=True
                )

            if count % 50000 == 0:
                gc.collect()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREDICTION FINISHED")
print("=" * 70)

print(
    f"Matching output:\n{MATCHING_OUT}"
)

print(
    f"Candidate output:\n{CANDIDATE_OUT}"
)

print(
    f"S1 processed: "
    f"{len(test_s1_records):,}"
)

print(
    "\nNext step: run the official validator."
)