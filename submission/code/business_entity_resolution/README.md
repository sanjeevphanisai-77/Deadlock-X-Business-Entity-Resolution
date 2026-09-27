# Deadlock-X - Business Entity Resolution

## 1. Project Overview

This project addresses the Business Entity Resolution Challenge. The objective is to identify records in Source 2 and Source 3 that correspond to entities in the deduplicated Source 1 reference dataset.

The solution processes business names, business addresses, and country information to generate reliable entity matches while avoiding exhaustive comparison between all records.

## 2. Problem Statement

The challenge provides three independent sources containing business records.

- Source 1: Deduplicated reference entities
- Source 2: Independent business records
- Source 3: Independent business records

For every Source 1 entity in the test dataset, the system identifies corresponding records from Source 2 and Source 3.

The input records may contain:

- Different capitalization
- Punctuation differences
- Business-name formatting differences
- Address formatting differences
- Abbreviations
- Whitespace variations
- Missing information
- Multiple records corresponding to the same business entity

## 3. Dataset Processing

All datasets are TSV files and are processed using tab-separated parsing.

The pipeline uses the following fields:

- entity_id
- business_name
- business_address
- country

Country is retained throughout processing and is used as part of the blocking key.

## 4. Data Normalization

Before matching, business names and addresses are normalized.

The normalization process performs:

1. Conversion to lowercase.
2. Replacement of non-alphanumeric characters with spaces.
3. Removal of unnecessary punctuation.
4. Collapsing consecutive whitespace.
5. Removal of leading and trailing whitespace.

Example:

Original:

    Prime-Money, Inc.

Normalized:

    prime money inc

The same normalization procedure is applied consistently to Source 1, Source 2, and Source 3.

## 5. Blocking / Candidate Generation

A full comparison between every Source 1 record and every Source 2/Source 3 record would be computationally expensive.

Therefore, the solution uses deterministic blocking.

Two indexes are created for Source 2 and Source 3:

### Name Blocking

    (country, normalized_business_name)

### Address Blocking

    (country, normalized_business_address)

For every Source 1 record, records sharing either the normalized business name or normalized business address within the same country are retrieved.

The candidate sets from both blocking strategies are combined.

This significantly reduces unnecessary comparisons.

## 6. Matching Strategy

The final fast pipeline uses deterministic matching.

A Source 2 or Source 3 record is considered a match candidate when:

- The country is the same, and
- The normalized business name matches exactly, OR
- The normalized business address matches exactly.

The union of name-based and address-based matches is used.

This approach is deterministic and reproducible.

## 7. Feature Engineering

The final pipeline uses the following matching information:

- Country
- Normalized business name
- Normalized business address
- Exact normalized-name equality
- Exact normalized-address equality

The approach intentionally avoids external business databases or external web lookups.

## 8. Candidate Output

The final candidate set is written to:

    output/candidate_pairs.tsv

The final selected entity matches are written to:

    output/matching_results.tsv

Both files contain one row for every Source 1 test entity.

Multiple matched Source 2/Source 3 IDs are represented as comma-separated IDs according to the challenge output specification.

Entities without a detected match have an empty matched_entity_ids field.

## 9. Validation

The final outputs were checked using the official challenge validator.

Validation was performed both with the standard validation and with ID-existence checking enabled.

The final validation result was:

    PASS - no blocking issues found. Safe to submit.

The ID-existence validation also passed successfully.

## 10. Reproducibility

The main implementation is provided in:

    code/business_entity_resolution/src/FINAL_FAST.py

Required Python package:

    pandas

The solution can be executed using:

    python src/FINAL_FAST.py

## 11. External Data

No external business databases, web searches, or third-party entity enrichment services are used.

The solution operates only on the supplied challenge datasets.

## 12. Limitations

The final implementation prioritizes computational efficiency and deterministic matching.

Exact normalized name and address matching may not recover every possible semantic or heavily corrupted business-name variation. More advanced fuzzy matching, multilingual transliteration, and learned pair classification could potentially improve recall, but the final submitted pipeline was designed to produce a complete valid submission efficiently.

## 13. Team

Team Name: Deadlock-X

## 14. Submission Files

The submission contains:

    output/
        matching_results.tsv
        candidate_pairs.tsv

    code/
        business_entity_resolution/
            src/
            README.md
            requirements.txt

    Documentation_template.md
