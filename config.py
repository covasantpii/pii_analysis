"""
Configuration for PII Detection using spaCy
"""

SPACY_MODEL = "en_core_web_sm"

{
    "chunk_size": 1500
}


# SpaCy NER labels treated as PII
PII_ENTITY_LABELS = {
    "PERSON",   # names
    "GPE",      # cities, countries
    "LOC",      # locations
    "ORG",      # organizations
    "FAC",      # facilities
    "DATE",     # dates
    "TIME",     # times
}
REGEX_PATTERNS = {

    # FULL NAME → 2 or 3 capitalized tokens
    "FULL_NAME": r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",

    # Last 4 PAN/CC number
    "CARD_LAST4": r"X{10,}[0-9]{4}",

    # Ticket ID → 15–20 digits
    "TICKET_ID": r"\b\d{15,20}\b",

    # Authorization Code → 5–8 digits
    "AUTHORIZATION": r"\b\d{5,8}\b",

    # Date format
    "DATE": r"\b\d{2}\/\d{2}\/\d{4}\b",

    # Time format
    "TIME": r"(0?[1-9]|1[0-2]):[0-5][0-9]\s?(AM|PM)",

    # Phone Number
    "PHONE": r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b",

    # Price amounts → decimals
    "PRICE": r"\b\d+\.\d{2}\b",
}


# OCR input / output defaults
DEFAULT_OCR_JSON = "data/Vamp0000040600_ocr.json"
OUTPUT_DIR = "data/output"
