import re
import unicodedata


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "inc", "incorporated",
    "llc", "ltd", "limited",
    "corp", "corporation",
    "co", "company",
    "plc", "pvt", "private",
    "llp", "gmbh", "ag", "sa", "spa"
}

ADDRESS_GENERIC = {
    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "lane", "ln",
    "drive", "dr",
    "boulevard", "blvd",
    "highway", "hwy",
    "way", "parkway", "pkwy",
    "place", "pl",
    "court", "ct",
    "circle", "cir",
    "floor", "fl",
    "building", "bldg",
    "unit", "suite", "ste"
}


def unicode_normalize(value):
    """
    Preserve Unicode characters while normalizing
    accents and compatibility characters.
    """
    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def ascii_normalize(value):
    """
    Latin/ASCII representation used for typo matching.
    """
    value = unicode_normalize(value)

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = value.encode(
        "ascii",
        "ignore"
    ).decode(
        "ascii"
    )

    value = value.replace(
        "&",
        " and "
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value
    )

    return " ".join(value.split())


def name_tokens(value):
    text = ascii_normalize(value)

    return {
        token
        for token in text.split()
        if len(token) >= 3
        and token not in LEGAL_SUFFIXES
    }


def address_tokens(value):
    text = ascii_normalize(value)

    return {
        token
        for token in text.split()
        if len(token) >= 3
        and token not in ADDRESS_GENERIC
    }


def number_tokens(value):
    text = ascii_normalize(value)

    return {
        token
        for token in text.split()
        if any(
            character.isdigit()
            for character in token
        )
    }


# ============================================================
# BASIC SIMILARITY FUNCTIONS
# ============================================================

def jaccard(set_a, set_b):

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def overlap_score(set_a, set_b):

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / min(
        len(set_a),
        len(set_b)
    )


def sequence_similarity(a, b):
    """
    Simple character similarity using
    Python's built-in SequenceMatcher.
    """

    from difflib import SequenceMatcher

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# NAME FEATURES
# ============================================================

def name_features(name_a, name_b):

    a_unicode = unicode_normalize(name_a)
    b_unicode = unicode_normalize(name_b)

    a_ascii = ascii_normalize(name_a)
    b_ascii = ascii_normalize(name_b)

    tokens_a = name_tokens(name_a)
    tokens_b = name_tokens(name_b)

    return {

        # Exact normalized representations
        "name_exact_unicode":
            int(a_unicode == b_unicode and bool(a_unicode)),

        "name_exact_ascii":
            int(a_ascii == b_ascii and bool(a_ascii)),

        # Token based
        "name_jaccard":
            jaccard(tokens_a, tokens_b),

        "name_overlap":
            overlap_score(tokens_a, tokens_b),

        # Character similarity
        "name_char_similarity":
            sequence_similarity(
                a_ascii,
                b_ascii
            ),

        # Length information
        "name_length_a":
            len(a_ascii),

        "name_length_b":
            len(b_ascii),

        "name_length_difference":
            abs(
                len(a_ascii) -
                len(b_ascii)
            )
    }


# ============================================================
# ADDRESS FEATURES
# ============================================================

def address_features(address_a, address_b):

    a_unicode = unicode_normalize(address_a)
    b_unicode = unicode_normalize(address_b)

    a_ascii = ascii_normalize(address_a)
    b_ascii = ascii_normalize(address_b)

    tokens_a = address_tokens(address_a)
    tokens_b = address_tokens(address_b)

    numbers_a = number_tokens(address_a)
    numbers_b = number_tokens(address_b)

    return {

        "address_exact_unicode":
            int(
                a_unicode == b_unicode
                and bool(a_unicode)
            ),

        "address_exact_ascii":
            int(
                a_ascii == b_ascii
                and bool(a_ascii)
            ),

        "address_jaccard":
            jaccard(
                tokens_a,
                tokens_b
            ),

        "address_overlap":
            overlap_score(
                tokens_a,
                tokens_b
            ),

        "address_char_similarity":
            sequence_similarity(
                a_ascii,
                b_ascii
            ),

        "address_number_overlap":
            overlap_score(
                numbers_a,
                numbers_b
            ),

        "address_length_difference":
            abs(
                len(a_ascii) -
                len(b_ascii)
            )
    }


# ============================================================
# COUNTRY FEATURE
# ============================================================

def country_feature(country_a, country_b):

    a = unicode_normalize(country_a)
    b = unicode_normalize(country_b)

    return int(
        bool(a)
        and bool(b)
        and a == b
    )


# ============================================================
# COMPLETE FEATURE VECTOR
# ============================================================

def match_features(
    name_a,
    address_a,
    country_a,
    name_b,
    address_b,
    country_b
):

    features = {}

    features.update(
        name_features(
            name_a,
            name_b
        )
    )

    features.update(
        address_features(
            address_a,
            address_b
        )
    )

    features["country_match"] = (
        country_feature(
            country_a,
            country_b
        )
    )

    return features


# ============================================================
# TEST CASES
# ============================================================

examples = [

    {
        "s1_name":
            "Ak Software Limited",

        "s1_address":
            "1-11-126/1/1, Shyamlal Building, Begumpet, Hyderabad, Telangana",

        "s1_country":
            "India",

        "s2_name":
            "AK S0FTWARE LIMITED",

        "s2_address":
            "",

        "s2_country":
            "India"
    },

    {
        "s1_name":
            "Indo Ventures Private Limited",

        "s1_address":
            "B 484, Nk-3, Indirapuram, Ghaziabad, Uttar Pradesh",

        "s1_country":
            "India",

        "s2_name":
            "इंडो वेंचर्स प्राइवेट लिमिटेड",

        "s2_address":
            "B 48, GAUTAM BUDDHA NAGAR, उत्तर प्रदेश",

        "s2_country":
            "India"
    },

    {
        "s1_name":
            "Mb Lifestyle Pvt Ltd",

        "s1_address":
            "West Bengal, Kolkata, Howrah, Rajarhat 20 Baijnathpur",

        "s1_country":
            "India",

        "s2_name":
            "Mb Pvt Ltd Center",

        "s2_address":
            "",

        "s2_country":
            "India"
    },

    {
        "s1_name":
            "G X & O Rothschild",

        "s1_address":
            "1001 Rivian Drive, Bloomington, IL",

        "s1_country":
            "US",

        "s2_name":
            "G X + O Rotshclhd",

        "s2_address":
            "",

        "s2_country":
            "US"
    },

    {
        "s1_name":
            "O+ Maritime",

        "s1_address":
            "4610 Frankfort Drive, Rockville, MD",

        "s1_country":
            "US",

        "s2_name":
            "O+ Service",

        "s2_address":
            "",

        "s2_country":
            "US"
    },

    {
        "s1_name":
            "BA Academy Pvt Ltd",

        "s1_address":
            "C/O Pandharinath Sawant, Hn - Hirlok Parabwadi, Kudal, Sindhudurg, Maharashtra",

        "s1_country":
            "India",

        "s2_name":
            "BA Pvt Ltd Center",

        "s2_address":
            "",

        "s2_country":
            "India"
    }
]


# ============================================================
# DISPLAY
# ============================================================

print("=" * 90)
print("MATCH FEATURE TEST")
print("=" * 90)

for i, example in enumerate(
    examples,
    start=1
):

    features = match_features(
        example["s1_name"],
        example["s1_address"],
        example["s1_country"],

        example["s2_name"],
        example["s2_address"],
        example["s2_country"]
    )

    print("\n" + "-" * 90)
    print(f"CASE {i}")

    print(
        f"S1: {example['s1_name']}"
    )

    print(
        f"S2: {example['s2_name']}"
    )

    print("\nFeatures:")

    for key, value in features.items():

        print(
            f"{key:30s}: {value}"
        )

print("\n" + "=" * 90)