import json
import re
import argparse
from pathlib import Path
import spacy

from config import (
    SPACY_MODEL,
    PII_ENTITY_LABELS,   # e.g. {"PERSON","ORG","GPE","NORP","DATE","TIME","LOC"}
    DEFAULT_OCR_JSON,
    OUTPUT_DIR
)

HEADER_MAX_LEN = 40  # max length of a "header" line

# ===================== Utility =====================

def extract_text_from_json(json_file: Path) -> str:
    """Extract combined text from OCR JSON {page_texts: {...}}."""
    data = json.loads(json_file.read_text(encoding="utf-8"))
    pages = data.get("page_texts", {})

    if isinstance(pages, dict):
        return "\n".join(pages.values())
    if isinstance(pages, list):
        return "\n".join(pages)
    return str(pages)


def split_text_into_chunks(text: str, chunk_size: int):
    """Yield (chunk_text, (start_offset, end_offset)) for large texts."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end], (start, end)
        start = end

# ===================== Validators =====================

def check_luhn(card_number: str) -> bool:
    """Return True if card_number passes Luhn checksum."""
    digits = re.sub(r"[^\d]", "", card_number)
    if not 12 <= len(digits) <= 19:
        return False

    total = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        n = int(d)
        # double every second digit (1-based from the right)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


def is_valid_phone_number(value: str) -> bool:
    """US-style 10-digit phone with NXX-NXX-XXXX constraints."""
    digits = re.sub(r"[^\d]", "", value)
    if len(digits) != 10:
        return False
    match = re.match(r"^([2-9]\d{2})([2-9]\d{2})(\d{4})$", digits)
    return match is not None

# ===================== Fallbacks (disabled) =====================

def extract_missing_person_names(_text: str):
    """Disabled: rely on spaCy for PERSON to avoid overfitting."""
    return []


def extract_missing_org_names(_text: str):
    """Disabled: rely on spaCy for ORG to avoid overfitting."""
    return []

# ===================== spaCy NER =====================

def extract_spacy_entities(nlp, text: str):
    """
    Use spaCy NER to get PERSON, ORG, and other configured entity labels.
    Apply only light, generic cleanup to reduce obvious noise.
    """
    doc = nlp(text)
    spans = []

    # Generic labels to drop (numbers, money, etc.)
    DISALLOWED_SPACY_LABELS = {
        "CARDINAL", "ORDINAL", "QUANTITY", "PERCENT", "MONEY",
        "LANGUAGE", "PRODUCT", "EVENT", "LAW"
    }

    # Very small deny-list for obvious UI / generic tokens
    UI_TOKENS = {
        "groups", "group", "settings", "home", "documents", "pages",
        "site", "contents", "recent", "title", "department", "search",
        "actions", "links", "view", "detail", "details", "about",
        "preview", "total", "subtotal", "tax", "tip"
    }

    # Specific bad full texts for ORG
    BLOCK_ORG_TEXTS = {"Sub Total", "Total", "Tax", "Tip"}

    for ent in doc.ents:
        txt = ent.text.strip()
        label = ent.label_
        words = txt.split()

        # Only keep labels your config says are interesting
        if label not in PII_ENTITY_LABELS:
            continue

        if label in DISALLOWED_SPACY_LABELS:
            continue

        if label == "ORG":
            if txt in BLOCK_ORG_TEXTS:
                continue
            if any(w.lower() in UI_TOKENS for w in words):
                continue

        if label == "PERSON":
            # discard extremely short / weird spans
            if len(txt) < 2:
                continue
            if "\n" in txt:
                continue
            if any(w.lower() in UI_TOKENS for w in words):
                continue

        spans.append({
            "pii_type": label,
            "text": txt,
            "start": ent.start_char,
            "end": ent.end_char,
            "source": "spacy"
        })

    return spans

# ===================== Regex PII =====================

def extract_regex_pii(text: str):
    """
    Regex-based detection for structured PII:
    SSN, DATE, TIME, PHONE, EMAIL, CREDIT_CARD, CARD_LAST4, DLN, AUTH_CODE, ZIPCODE.
    """
    spans = []

    patterns = {
        "SSN":         r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
        "DATE":        r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-]\d{4}\b",
        "TIME":        r"\b(0?[1-9]|1[0-2]):[0-5]\d\s?(?:AM|PM)\b",
        "PHONE":       r"\b[0-9]{3}[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b",
        "EMAIL":       r"[A-Za-z0-9._%+-]+@[A-Za-z0-9._-]+\.[A-Za-z]{2,}",
        "CREDIT_CARD": r"\b(?:\d[ -]*){12,19}\b",
        "CARD_LAST4":  r"\b[Xx]{4,}\d{4}\b",
        "DLN":         r"\b(?:DLN|DL#|DL:|Driver|License)[\s:]*([A-Za-z]?\d{8,12})\b",
        "AUTH_CODE":   r"\b\d{6,12}\b",
        "ZIPCODE":     r"\b\d{5}\b"
    }

    for base_label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            value = match.group().strip()
            digits = re.sub(r"[^\d]", "", value)
            label = base_label

            if label == "PHONE":
                if not is_valid_phone_number(value):
                    # reclassify 9-digit invalid phone as SSN
                    if len(digits) == 9:
                        label = "SSN"
                    else:
                        continue

            if label == "ZIPCODE":
                # filter clearly invalid ZIPs, like 00000 or 000xx
                if digits.startswith("000") or digits == "00000":
                    continue

            if label == "CREDIT_CARD" and not check_luhn(value):
                continue

            spans.append({
                "pii_type": label,
                "text": value,
                "start": match.start(),
                "end": match.end(),
                "source": "regex"
            })

    return spans

# ===================== Header Association =====================

def build_line_index(text: str):
    """
    Build:
    - lines: list of {"start": s, "end": e, "text": line_text}
    - char_to_line: list mapping each char index -> line index
    """
    lines = []
    char_to_line = [0] * len(text) if text else []
    offset = 0
    for idx, raw in enumerate(text.splitlines(keepends=True)):
        line_text = raw.rstrip("\n\r")
        start = offset
        end = start + len(line_text)
        lines.append({"start": start, "end": end, "text": line_text})
        for pos in range(start, end):
            if pos < len(char_to_line):
                char_to_line[pos] = idx
        offset += len(raw)
    return lines, char_to_line


def is_header_candidate(line_text: str) -> bool:
    """
    Heuristic to decide if a line looks like a header:
    - Short line
    - Mostly alphanumeric + simple punctuation
    - Often ends with ':' or '#', or is 1–3 tokens with no digits
    """
    stripped = line_text.strip()
    if not stripped or len(stripped) > HEADER_MAX_LEN:
        return False

    # mostly letters / digits / spaces / simple punctuation
    if not re.match(r"^[A-Za-z0-9 #:/\-\(\)]+$", stripped):
        return False

    # strong signal if ends with ':' or '#'
    if stripped.endswith((":","#")):
        return True

    # otherwise, if 1–3 words and no digits inside words → treat as header
    tokens = stripped.split()
    if 1 <= len(tokens) <= 3:
        if not any(any(c.isdigit() for c in t) for t in tokens):
            return True

    return False


def attach_headers_to_pii(text: str, pii_spans: list):
    """
    For each PII span, find nearest header above/below and add 'header' key.
    Logic:
      - Build per-line index
      - Mark lines that look like headers
      - For each PII span: look at header in same line, line-1, line+1
    """
    if not text or not pii_spans:
        return pii_spans

    lines, char_to_line = build_line_index(text)

    # precompute header candidates per line
    header_by_line = {}
    for i, line in enumerate(lines):
        if is_header_candidate(line["text"]):
            header_by_line[i] = line["text"].strip()

    for span in pii_spans:
        start = span["start"]
        if not char_to_line or start >= len(char_to_line):
            continue

        line_idx = char_to_line[start]
        candidate_headers = []

        # same line
        if line_idx in header_by_line:
            candidate_headers.append((0, header_by_line[line_idx]))

        # line above
        if line_idx - 1 in header_by_line:
            candidate_headers.append((1, header_by_line[line_idx - 1]))

        # line below
        if line_idx + 1 in header_by_line:
            candidate_headers.append((1, header_by_line[line_idx + 1]))

        if candidate_headers:
            # pick closest (distance), prefer same line if exists
            candidate_headers.sort(key=lambda x: x[0])
            span["header"] = candidate_headers[0][1]

    return pii_spans

# ===================== Conflict Resolution =====================

def resolve_conflicts(spans):
    """
    If the same text appears with multiple labels, keep the highest-priority one.
    Priority is defined by domain knowledge.
    """
    priority = [
        "SSN", "CREDIT_CARD", "DLN", "EMAIL", "PHONE",
        "ZIPCODE", "DATE", "TIME",
        "CARD_LAST4", "AUTH_CODE",
        "PERSON", "ORG", "GPE", "NORP", "URL"
    ]

    def get_priority(label: str) -> int:
        return priority.index(label) if label in priority else len(priority)

    final = []
    seen = {}

    for item in spans:
        txt = item["text"]
        if txt in seen:
            prev_label = seen[txt]
            if get_priority(item["pii_type"]) < get_priority(prev_label):
                # replace older, lower-priority entry
                final = [x for x in final if x["text"] != txt]
                final.append(item)
                seen[txt] = item["pii_type"]
        else:
            final.append(item)
            seen[txt] = item["pii_type"]

    return final

# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_OCR_JSON)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    json_path = Path(args.input)
    try:
        conf = json.loads(Path("app_config.json").read_text(encoding="utf-8"))
        chunk_size = conf.get("chunk_size", args.chunk_size)
    except Exception:
        chunk_size = args.chunk_size

    if not chunk_size:
        chunk_size = 7000

    text = extract_text_from_json(json_path)
    nlp = spacy.load(SPACY_MODEL)

    all_pii = []
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    for idx, (chunk, (start, end)) in enumerate(split_text_into_chunks(text, chunk_size)):
        sp_hits = extract_spacy_entities(nlp, chunk)      # PERSON / ORG / others
        regex_hits = extract_regex_pii(chunk)              # phone, card, date, etc.
        # fallbacks currently disabled
        fallback_person_hits = extract_missing_person_names(chunk)
        fallback_org_hits = extract_missing_org_names(chunk)

        merged = []
        for hit in sp_hits + regex_hits + fallback_person_hits + fallback_org_hits:
            new = dict(hit)
            new["start"] += start
            new["end"] += start
            new["chunk"] = idx
            merged.append(new)

        cleaned = resolve_conflicts(merged)
        all_pii.extend(cleaned)

    # Attach nearest headers (same/above/below line) to each PII span
    all_pii = attach_headers_to_pii(text, all_pii)

    output_file = Path(args.output_dir) / "output_PII.json"
    output_file.write_text(json.dumps(all_pii, indent=2), encoding="utf-8")
    print("✔ DONE")


if __name__ == "__main__":
    main()
