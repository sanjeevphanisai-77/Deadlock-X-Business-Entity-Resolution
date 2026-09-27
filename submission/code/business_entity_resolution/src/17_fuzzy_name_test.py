import re


def normalize_name(value):
    if value is None:
        return ""

    value = str(value).lower()

    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def char_ngrams(text, n=3):
    text = text.replace(" ", "")

    if len(text) < n:
        return {text}

    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


def jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def similarity(a, b):

    a = normalize_name(a)
    b = normalize_name(b)

    if not a or not b:
        return 0.0

    grams_a = char_ngrams(a)
    grams_b = char_ngrams(b)

    return jaccard(grams_a, grams_b)


# ============================================================
# MISSED MATCHES
# ============================================================

examples = [

    (
        "Ak Software Limited",
        "AK S0FTWARE LIMITED"
    ),

    (
        "Indo Ventures Private Limited",
        "इंडो वेंचर्स प्राइवेट लिमिटेड"
    ),

    (
        "Mb Lifestyle Pvt Ltd",
        "Mb Pvt Ltd Center"
    ),

    (
        "G X & O Rothschild",
        "G X + O Rotshclhd"
    ),

    (
        "O+ Maritime",
        "O+ Service"
    ),

    (
        "BA Academy Pvt Ltd",
        "BA Pvt Ltd Center"
    )
]


print("=" * 70)
print("FUZZY NAME TEST")
print("=" * 70)

for s1, s2 in examples:

    score = similarity(s1, s2)

    print("\nS1:", s1)
    print("S2:", s2)
    print(f"Character 3-gram Jaccard: {score:.4f}")

print("\n" + "=" * 70)