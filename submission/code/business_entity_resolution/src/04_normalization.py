import re
import unicodedata


# =========================================================
# Unicode normalization
# =========================================================

def normalize_unicode(text):
    """
    Normalize Unicode while preserving meaningful characters.
    """
    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)

    return text


# =========================================================
# Basic text normalization
# =========================================================

def normalize_text(text):
    """
    General normalization for business names and addresses.
    """

    if text is None:
        return ""

    text = normalize_unicode(text)

    # Lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# Alphanumeric representation
# =========================================================

def normalize_alphanumeric(text):
    """
    Remove spaces and punctuation.
    Useful for robust exact/blocking comparisons.
    """

    text = normalize_text(text)

    return re.sub(r"[^\w]", "", text, flags=re.UNICODE)


# =========================================================
# Token representation
# =========================================================

def tokenize(text):
    """
    Convert text into normalized tokens.
    """

    text = normalize_text(text)

    if not text:
        return []

    return text.split()


def sorted_token_string(text):
    """
    Sort tokens alphabetically.

    Useful when word order changes.
    """

    tokens = tokenize(text)

    return " ".join(sorted(tokens))


# =========================================================
# Character n-grams
# =========================================================

def char_ngrams(text, n=3):
    """
    Generate character n-grams.
    """

    text = normalize_alphanumeric(text)

    if len(text) < n:
        return {text} if text else set()

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


# =========================================================
# Simple abbreviation normalization
# =========================================================

ABBREVIATIONS = {
    "street": "st",
    "st.": "st",
    "road": "rd",
    "rd.": "rd",
    "avenue": "ave",
    "ave.": "ave",
    "boulevard": "blvd",
    "blvd.": "blvd",
    "drive": "dr",
    "dr.": "dr",
    "lane": "ln",
    "ln.": "ln",
    "highway": "hwy",
    "hwy.": "hwy",
    "parkway": "pkwy",
    "pkwy.": "pkwy",
    "apartment": "apt",
    "apt.": "apt",
    "suite": "ste",
    "ste.": "ste",
}


def normalize_address_tokens(text):
    """
    Normalize common English address terms.
    """

    tokens = tokenize(text)

    normalized = [
        ABBREVIATIONS.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized)


# =========================================================
# Business name suffix normalization
# =========================================================

LEGAL_SUFFIXES = {
    "incorporated": "inc",
    "inc": "inc",
    "corporation": "corp",
    "corp": "corp",
    "limited": "ltd",
    "ltd": "ltd",
    "llc": "llc",
    "private": "pvt",
    "pvt": "pvt",
}


def normalize_business_name(text):
    """
    Normalize business name while retaining legal suffixes.
    """

    tokens = tokenize(text)

    normalized = [
        LEGAL_SUFFIXES.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized)


# =========================================================
# Demonstration
# =========================================================

if __name__ == "__main__":

    examples = [
        "Prime Money, Inc.",
        "PRIME MONEY INC",
        "17560 Ellis Road, Tahlequah, OK",
        "17560 ELLIS RD TAHLEQUAH OK",
        "B+ Retail Inc.",
        "Christ Chapel - Apartment G",
    ]

    print("=" * 70)
    print("NORMALIZATION TEST")
    print("=" * 70)

    for value in examples:

        print(f"\nOriginal : {value}")
        print(f"Normal   : {normalize_text(value)}")
        print(f"Alnum    : {normalize_alphanumeric(value)}")
        print(f"Tokens   : {tokenize(value)}")
        print(f"Sorted   : {sorted_token_string(value)}")
        print(f"3-grams  : {list(char_ngrams(value))[:10]}")

    print("\n" + "=" * 70)
    print("NORMALIZATION TEST COMPLETE")
    print("=" * 70)