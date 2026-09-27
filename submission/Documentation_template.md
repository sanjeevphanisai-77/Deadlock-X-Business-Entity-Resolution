# Deadlock-X
# Business Entity Resolution Challenge
# Methodology Documentation

## 1. Introduction

The objective of this project is to resolve business entities across multiple independent data sources.

Source 1 acts as the deduplicated reference dataset. Source 2 and Source 3 contain independent records that may correspond to the entities present in Source 1.

The system identifies Source 2 and Source 3 records that correspond to each Source 1 entity.

The solution was designed with emphasis on:

- Data normalization
- Country-aware blocking
- Efficient candidate generation
- Deterministic matching
- Valid submission generation
- Reproducibility

---

# 2. Input Data

The challenge provides training and test datasets containing three business-record sources.

Each source contains the following primary fields:

- entity_id
- business_name
- business_address
- country

The training data additionally provides ground-truth relationships between Source 1 entities and their corresponding Source 2/Source 3 records.

The final prediction stage operates on the supplied test datasets.

---

# 3. Data Characteristics

Business records can differ even when they represent the same real-world entity.

Examples of variation include:

- Uppercase/lowercase differences
- Punctuation differences
- Different spacing
- Business-name formatting differences
- Abbreviations
- Address formatting differences
- Reordering of address components
- Missing address components
- Typographical differences
- Different representations of business information

Because of these variations, raw string equality is insufficient for general entity resolution.

The final fast pipeline therefore applies consistent normalization before performing exact blocking and matching.

---

# 4. Data Normalization

The same normalization function is applied to business names and addresses.

The normalization procedure consists of:

1. Convert text to lowercase.
2. Replace non-alphanumeric characters with spaces.
3. Collapse consecutive whitespace.
4. Remove leading and trailing whitespace.

For example:

Input:

    PRIME-MONEY, INC.

Normalized:

    prime money inc

Another example:

Input:

    123 Main-Street, New York

Normalized:

    123 main street new york

This allows formatting differences to be ignored during exact normalized matching.

---

# 5. Blocking Strategy

Directly comparing every Source 1 record against every Source 2 and Source 3 record would create an extremely large number of pairwise comparisons.

To reduce this search space, the solution uses blocking.

Two independent blocking indexes are constructed.

## 5.1 Name-Based Blocking

The first index uses:

    country + normalized business name

A Source 1 record retrieves Source 2/Source 3 records with the same country and normalized business name.

## 5.2 Address-Based Blocking

The second index uses:

    country + normalized business address

A Source 1 record retrieves Source 2/Source 3 records with the same country and normalized business address.

## 5.3 Candidate Union

The two candidate sets are combined.

Therefore a candidate can be retrieved through either:

    Name match

or

    Address match

while still respecting the country constraint.

This provides two independent opportunities for recovering a corresponding business record.

---

# 6. Country-Aware Matching

Country is included in the blocking key.

This prevents records with identical normalized names or addresses from unrelated countries from being unnecessarily treated as matches.

The implementation does not hard-code a particular set of countries.

The country value supplied by the dataset is used directly.

---

# 7. Matching Procedure

For each Source 1 test entity:

1. Normalize its business name.
2. Normalize its business address.
3. Read its country.
4. Retrieve records from the name index.
5. Retrieve records from the address index.
6. Combine the retrieved records.
7. Remove duplicate candidate IDs.
8. Write the resulting Source 2/Source 3 IDs as the entity's matches.

The final output therefore contains all matches obtained by the deterministic blocking rules.

---

# 8. Feature Engineering

The final deterministic pipeline uses the following information:

| Feature | Purpose |
|---|---|
| Country | Country-aware blocking |
| Normalized business name | Name-based matching |
| Normalized business address | Address-based matching |
| Name equality | Determines exact normalized-name correspondence |
| Address equality | Determines exact normalized-address correspondence |

The implementation does not use external business attributes.

---

# 9. Candidate Pair Generation

The challenge requires candidate_pairs.tsv to represent the final candidate set used immediately before matching.

The final pipeline produces the candidate set from the same deterministic matching process.

The candidate output contains:

- Source 1 entity ID
- Candidate Source 2/Source 3 entity IDs

Every selected match is therefore represented in the candidate output.

---

# 10. Final Output Generation

Two files are generated.

## matching_results.tsv

This file contains the final entity-resolution results.

Each Source 1 test entity appears exactly once.

The output contains:

- source1_entity_id
- matched_entity_ids

If no corresponding record is identified, the matched entity list is empty.

## candidate_pairs.tsv

This file contains the final candidate relationships produced by the candidate-generation stage.

The candidate file is generated consistently with the final matching output.

---

# 11. Computational Efficiency

The datasets contain millions of records.

An exhaustive pairwise comparison would be computationally expensive.

The use of dictionary-based indexes allows records to be retrieved directly using normalized keys instead of comparing each Source 1 record with every Source 2 and Source 3 record.

The main indexes are:

    (country, normalized_name)

and:

    (country, normalized_address)

This significantly reduces the amount of unnecessary comparison.

---

# 12. External Data Policy

No external data lookup is performed.

The solution uses only the datasets supplied as part of the challenge.

No:

- Search engine lookup
- Business directory lookup
- External API enrichment
- External company database

is used.

---

# 13. Reproducibility

The implementation is provided at:

    code/business_entity_resolution/src/FINAL_FAST.py

The required dependency is:

    pandas

The pipeline can be executed using:

    python src/FINAL_FAST.py

The generated output files are:

    output/matching_results.tsv
    output/candidate_pairs.tsv

---

# 14. Validation

The final submission was tested using the organizer-provided validation utility.

Validation was performed using:

    python submission/validate_submission.py \
        --matching submission/output/matching_results.tsv \
        --candidate submission/output/candidate_pairs.tsv \
        --test-dir dataset/test \
        --check-ids

The final validation result was:

    PASS - no blocking issues found. Safe to submit.

The validator confirmed that:

- The required number of Source 1 entities is present.
- The matching output has the required number of rows.
- The candidate output has the required number of rows.
- Matched IDs exist in the supplied test Source 2/Source 3 data.
- No blocking submission-format violations were detected.

---

# 15. Final Submission Statistics

The final test Source 1 dataset contains:

    1,732,544

Source 1 entities.

The generated matching output contains:

    1,732,544

rows.

The generated candidate output contains:

    1,732,544

rows.

The validator reported:

    393,365 empty records
    1,339,179 non-empty records

The validator reported:

    valid S2/S3 match IDs: 9,969,589

---

# 16. Limitations

The final deterministic pipeline prioritizes speed, reproducibility, and complete output generation.

Exact normalized name and address matching cannot capture every possible semantic variation.

For example, heavily corrupted names, transliterations, abbreviations, or records with both missing names and missing addresses may require more advanced techniques.

Potential future improvements include:

- Character-level similarity
- Token-level similarity
- Address number matching
- Fuzzy string matching
- Transliteration-aware matching
- Learned pairwise classification
- Confidence scoring
- Frequency-aware blocking

These techniques were explored during development, but the final submission uses the deterministic fast pipeline to ensure that the complete test dataset can be processed reliably within the available execution constraints.

---

# 17. Team

Team Name:

    Deadlock-X

---

# 18. Conclusion

The Deadlock-X solution provides a reproducible and computationally efficient approach to business entity resolution using normalized business names, normalized addresses, and country-aware blocking.

The final submission was successfully validated using the official challenge validator with ID checking enabled.
