import os
import re
import pandas as pd
from collections import defaultdict

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../..")
)

TEST = os.path.join(ROOT, "dataset", "test")
OUT = os.path.join(ROOT, "submission", "output")

S1 = os.path.join(TEST, "test_source1.tsv")
S2 = os.path.join(TEST, "test_source2.tsv")
S3 = os.path.join(TEST, "test_source3.tsv")

MATCH = os.path.join(OUT, "matching_results.tsv")
CAND = os.path.join(OUT, "candidate_pairs.tsv")

os.makedirs(OUT, exist_ok=True)


def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_index(path, label):
    print(f"Loading {label}...")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )

    df["name_n"] = df["business_name"].map(norm)
    df["addr_n"] = df["business_address"].map(norm)
    df["country_n"] = df["country"].fillna("").str.lower()

    name_index = defaultdict(list)
    addr_index = defaultdict(list)

    for row in df.itertuples(index=False):

        eid = row.entity_id
        country = row.country_n

        if row.name_n:
            name_index[
                (country, row.name_n)
            ].append(eid)

        if row.addr_n:
            addr_index[
                (country, row.addr_n)
            ].append(eid)

    print(
        f"{label}: {len(df):,} rows | "
        f"name keys={len(name_index):,} | "
        f"address keys={len(addr_index):,}"
    )

    del df

    return name_index, addr_index


print("=" * 70)
print("FINAL FAST ENTITY RESOLUTION")
print("=" * 70)

# ------------------------------------------------------------
# BUILD S2 INDEX
# ------------------------------------------------------------

s2_name, s2_addr = load_index(
    S2,
    "TEST Source 2"
)

# ------------------------------------------------------------
# BUILD S3 INDEX
# ------------------------------------------------------------

s3_name, s3_addr = load_index(
    S3,
    "TEST Source 3"
)

# ------------------------------------------------------------
# LOAD S1
# ------------------------------------------------------------

print("\nLoading TEST Source 1...")

s1 = pd.read_csv(
    S1,
    sep="\t",
    dtype=str
)

print(
    f"Test Source 1: {len(s1):,}"
)

# ------------------------------------------------------------
# OVERWRITE OLD PARTIAL FILES
# ------------------------------------------------------------

with open(
    MATCH,
    "w",
    encoding="utf-8",
    newline=""
) as mf, open(
    CAND,
    "w",
    encoding="utf-8",
    newline=""
) as cf:

    mf.write(
        "source1_entity_id\tmatched_entity_ids\n"
    )

    cf.write(
        "source1_entity_id\tcandidate_entity_ids\n"
    )

    total = len(s1)

    for count, row in enumerate(
        s1.itertuples(index=False),
        1
    ):

        eid = row.entity_id

        country = (
            str(row.country).lower()
            if not pd.isna(row.country)
            else ""
        )

        name = norm(row.business_name)
        addr = norm(row.business_address)

        candidates = []

        # ----------------------------------------------------
        # EXACT NORMALIZED NAME
        # ----------------------------------------------------

        if name:
            candidates.extend(
                s2_name.get(
                    (country, name),
                    []
                )
            )

            candidates.extend(
                s3_name.get(
                    (country, name),
                    []
                )
            )

        # ----------------------------------------------------
        # EXACT NORMALIZED ADDRESS
        # ----------------------------------------------------

        if addr:
            candidates.extend(
                s2_addr.get(
                    (country, addr),
                    []
                )
            )

            candidates.extend(
                s3_addr.get(
                    (country, addr),
                    []
                )
            )

        # Remove duplicates while preserving order
        candidates = list(
            dict.fromkeys(candidates)
        )

        # ----------------------------------------------------
        # FINAL MATCHES
        #
        # These are exactly the candidates considered.
        # ----------------------------------------------------

        matches = candidates

        cf.write(
            eid
            + "\t"
            + ",".join(candidates)
            + "\n"
        )

        mf.write(
            eid
            + "\t"
            + ",".join(matches)
            + "\n"
        )

        if count % 10000 == 0:

            print(
                f"Processed "
                f"{count:,}/{total:,} "
                f"({count / total * 100:.2f}%)",
                flush=True
            )

print("\n" + "=" * 70)
print("FINAL FAST GENERATION FINISHED")
print("=" * 70)

print(f"Matching:  {MATCH}")
print(f"Candidates: {CAND}")
print(f"S1 rows:   {len(s1):,}")