import json
import re
import argparse
from pathlib import Path
import spacy

from config import (
    SPACY_MODEL,
    PII_ENTITY_LABELS,
    DEFAULT_OCR_JSON,
    OUTPUT_DIR
)

# ===================== Utility =====================

def extract_text_from_json(json_file: Path) -> str:
    """Extract text from JSON OCR."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    pages = data.get("page_texts", {})

    if isinstance(pages, dict):
        return "\n".join(pages.values())

    if isinstance(pages, list):
        return "\n".join(pages)

    return str(pages)


def split_text_into_chunks(text: str, chunk_size: int):
    """Yield chunks based on size."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end], (start, end)
        start = end


# ===================== Validators =====================

def check_luhn(card_number: str) -> bool:
    """Credit card validation using Luhn algorithm."""
    digits = re.sub(r'[^\d]', '', card_number)

    if not 12 <= len(digits) <= 19:
        return False

    total, reverse_digits = 0, digits[::-1]

    for i, d in enumerate(reverse_digits):
        n = int(d)
        if i % 2 == 1:
            n = n * 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


# ===================== Extractors =====================

def extract_spacy_entities(nlp, text):
    doc = nlp(text)
    spans = []

    for ent in doc.ents:
        if ent.label_ in PII_ENTITY_LABELS:
            spans.append({
                "pii_type": ent.label_,
                "text": ent.text.strip(),
                "start": ent.start_char,
                "end": ent.end_char
            })
    return spans


def extract_regex_pii(text):
    spans = []

    patterns = {
        "SSN": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",

        "DOB": r"(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-](19\d\d|20[01]\d)",

        "DATE": r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-]\d{4}\b",

        "TIME": r"(0?[1-9]|1[0-2]):[0-5]\d\s?(AM|PM)",

        "PHONE": r"(?:\+?1[-.\s]?)?\(?[2-9]\d\d\)?[-.\s]?[2-9]\d\d[-.\s]?\d{4}",

        "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9._-]+\.[A-Za-z]{2,}",

        "CREDIT_CARD": r"\b(?:\d[ -]*){12,19}\b",

        "CARD_LAST4": r"\b[Xx]{6,}\d{4}\b",

        "DLN": r"(DLN|DL#|DL:|Driver|License)[\s:]*([A-Za-z]?\d{8,12})",

        "AUTH_CODE": r"\b\d{6,12}\b",

        "ZIPCODE": r"\b\d{5}(?:-\d{4})?\b"
    }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            value = match.group()

            # Validate card numbers
            if label == "CREDIT_CARD" and not check_luhn(value):
                continue  

            # Reject DLN that are actually invoices (too long)
            if label == "DLN" and value.isdigit() and len(value) > 12:
                continue  

            spans.append({
                "pii_type": label,
                "text": value.strip(),
                "start": match.start(),
                "end": match.end()
            })

    return spans


# ===================== Conflict Resolution =====================

def resolve_conflicts(spans):
    priority = [
        "SSN", "CREDIT_CARD", "DLN", "DOB",
        "EMAIL", "PHONE", "PERSON",
        "ZIPCODE", "DATE", "TIME",
        "CARD_LAST4", "AUTH_CODE", "URL", "ORG"
    ]

    final = []
    memory = {}

    for item in spans:
        txt = item["text"].strip()
        label = item["pii_type"]

        # CORRECTION: PERSON wrongly detecting masked card
        if label == "PERSON" and re.fullmatch(r"[Xx]{6,}\d{4}", txt):
            label = "CARD_LAST4"
            item["pii_type"] = label

        # CORRECTION: ORG wrongly detecting DATE
        if label == "ORG" and re.fullmatch(r"(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/]\d{4}", txt):
            label = "DATE"
            item["pii_type"] = label

        # Remove fake ORGs like:
        # TOTAL, TIP, TAX, INVOICE, numeric-only
        if label == "ORG":
            if txt.isupper() and len(txt) <= 4:
                continue
            if re.search(r"\d", txt):
                continue

        # Resolution based on priority
        if txt in memory:
            if priority.index(label) < priority.index(memory[txt]):
                final = [f for f in final if f["text"] != txt]
                final.append(item)
                memory[txt] = label
        else:
            final.append(item)
            memory[txt] = label

    return final


# ===================== MAIN =====================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_OCR_JSON)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    json_path = Path(args.input)

    # load config override
    try:
        conf = json.loads(Path("app_config.json").read_text())
        chunk_size = conf.get("chunk_size", args.chunk_size)
    except:
        chunk_size = args.chunk_size

    if not chunk_size:
        chunk_size = 7000

    print(f"Processing with CHUNK SIZE = {chunk_size}")

    text = extract_text_from_json(json_path)
    nlp = spacy.load(SPACY_MODEL)

    all_pii = []
    base_name = json_path.stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for idx, (chunk, (start, end)) in enumerate(split_text_into_chunks(text, chunk_size)):
        print(f"\n---- Chunk {idx+1} [{start}-{end}] ----")
        sp_hits = extract_spacy_entities(nlp, chunk)
        rg_hits = extract_regex_pii(chunk)

        merged = []
        for hit in sp_hits + rg_hits:
            merged.append({
                **hit,
                "start": hit["start"] + start,
                "end": hit["end"] + start,
                "chunk": idx
            })

        cleaned = resolve_conflicts(merged)
        all_pii.extend(cleaned)

    out_file = out_dir / f"{base_name}_PII.json"
    out_file.write_text(json.dumps(all_pii, indent=2), encoding="utf-8")

    print("\n✔ DONE ✔")
    print(f"PII COUNT = {len(all_pii)}")
    print("OUTPUT →", out_file)


if __name__ == "__main__":
    main()
