import json
import re
import argparse
from pathlib import Path
import spacy

# ===================== CONFIG (INLINE) =====================

# Use a model with NER enabled
SPACY_MODEL = "en_core_web_sm"

# Entity labels considered PII (tune as needed)
PII_ENTITY_LABELS = {
    "PERSON", "ORG", "GPE", "NORP", "DATE", "TIME", "LOC", "FAC"
}

# Defaults (override via CLI if desired)
DEFAULT_OCR_JSON = "ocr_input.json"
OUTPUT_DIR = "pii_output"

HEADER_MAX_LEN = 80  # allow reasonably long labels
ZIP_RE = re.compile(r"\b\d{5}\b")

# texts that must never be treated as headers
BAD_HEADER_TEXTS = {
    "Madam/ Dear Sir,",  # add more greetings / boilerplate if needed
}

# entity texts you do NOT want to use as grouping owners
BAD_ENTITY_KEYS = {
    "Madam/ Dear Sir,",
    "Request",
    "CIF",  # remove from here if you want CIF to own a group
}

# only these entity labels can own groups
GOOD_OWNER_LABELS = {"PERSON", "ORG"}

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
        if i % 2 == 1:  # double every second digit (1-based from the right)
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
    """Use spaCy NER to get configured entity labels and lightly clean noise."""
    doc = nlp(text)
    spans = []

    DISALLOWED_SPACY_LABELS = {
        "CARDINAL",
        "ORDINAL",
        "QUANTITY",
        "PERCENT",
        "MONEY",
        "LANGUAGE",
        "PRODUCT",
        "EVENT",
        "LAW",
    }

    UI_TOKENS = {
        "groups",
        "group",
        "settings",
        "home",
        "documents",
        "pages",
        "site",
        "contents",
        "recent",
        "title",
        "department",
        "search",
        "actions",
        "links",
        "view",
        "detail",
        "details",
        "about",
        "preview",
        "total",
        "subtotal",
        "tax",
        "tip",
    }

    BLOCK_ORG_TEXTS = {"Sub Total", "Total", "Tax", "Tip"}

    for ent in doc.ents:
        txt = ent.text.strip()
        label = ent.label_
        words = txt.split()

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
            if len(txt) < 2:
                continue
            if "\n" in txt:
                continue
            if any(w.lower() in UI_TOKENS for w in words):
                continue

        spans.append(
            {
                "pii_type": label,
                "text": txt,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy",
            }
        )

    return spans

# ===================== Regex PII =====================

def extract_regex_pii(text: str):
    """Regex-based detection for structured PII."""
    spans = []

    # extended patterns: standard plus long dates & currency
    patterns = {
        "SSN": r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
        "DATE": r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-]\d{4}\b",
        # long date (e.g. January 1, 2023)
        "DATE_LONG": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
        "TIME": r"\b(0?[1-9]|1[0-2]):[0-5]\d\s?(?:AM|PM)\b",
        "PHONE": r"\b[0-9]{3}[-\s]?[0-9]{3}[-\s]?[0-9]{4}\b",
        "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9._-]+\.[A-Za-z]{2,}",
        "CREDIT_CARD": r"\b(?:\d[ -]*){12,19}\b",
        "CARD_LAST4": r"\b[Xx]{4,}\d{4}\b",
        "DLN": r"\b(?:DLN|DL#|DL:|Driver|License)[\s:]*([A-Za-z]?\d{8,12})\b",
        "AUTH_CODE": r"\b\d{6,12}\b",
        "ZIPCODE": r"\b\d{5}\b",
        # amounts like $271.38 or $1,024.02 (treat as sensitive if desired)
        "CURRENCY": r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    }

    for base_label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            value = match.group().strip()
            digits = re.sub(r"[^\d]", "", value)
            label = base_label

            if label == "PHONE":
                if not is_valid_phone_number(value):
                    if len(digits) == 9:
                        label = "SSN"
                    else:
                        continue

            if label == "ZIPCODE":
                if digits.startswith("000") or digits == "00000":
                    continue

            if label == "CREDIT_CARD" and not check_luhn(value):
                continue

            # normalize DATE_LONG to DATE
            if label == "DATE_LONG":
                label = "DATE"

            spans.append(
                {
                    "pii_type": label,
                    "text": value,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "regex",
                }
            )

    return spans

# ===================== Header Association =====================

def build_line_index(text: str):
    """Build line list and char_to_line mapping."""
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
    """Heuristic to decide if a line looks like a header."""
    stripped = line_text.strip()
    if not stripped or len(stripped) > HEADER_MAX_LEN:
        return False

    if stripped in BAD_HEADER_TEXTS:
        return False

    if re.fullmatch(r"[0-9\-\s()+]+", stripped):
        return False

    if not re.match(r"^[A-Za-z0-9 #:/\-\(\)\.,]+$", stripped):
        return False

    if stripped.endswith((":", "#")):
        return True

    tokens = stripped.split()
    if 1 <= len(tokens) <= 6:
        return True

    return False


def normalize_header_key(header_text: str) -> str:
    """
    Reduce a full header line to just the header 'key'.
    """
    if header_text is None:
        return None
    stripped = header_text.strip()

    if ":" in stripped:
        parts = stripped.split(":", 1)
        key = parts[0].strip()
        return key or stripped

    m = ZIP_RE.search(stripped)
    if m and m.end() == len(stripped):
        before = stripped[: m.start()].rstrip(" ,")
        return before or stripped

    return stripped


def attach_headers_to_pii(text: str, pii_spans: list):
    """
    For each PII span, find nearest header above/below and add 'header' and 'line_idx'.
    """
    if not text or not pii_spans:
        return pii_spans

    lines, char_to_line = build_line_index(text)

    header_by_line = {}
    for i, line in enumerate(lines):
        if is_header_candidate(line["text"]):
            header_by_line[i] = line["text"].strip()

    for span in pii_spans:
        start = span["start"]
        if not char_to_line or start >= len(char_to_line):
            continue

        line_idx = char_to_line[start]
        span["line_idx"] = line_idx

        candidate_headers = []

        if line_idx in header_by_line:
            candidate_headers.append((0, header_by_line[line_idx]))

        for delta in (1, 2):
            up = line_idx - delta
            if up in header_by_line:
                candidate_headers.append((delta, header_by_line[up]))

        for delta in (1, 2):
            down = line_idx + delta
            if down in header_by_line:
                candidate_headers.append((delta, header_by_line[down]))

        if candidate_headers:
            candidate_headers.sort(key=lambda x: x[0])
            best_header_full = candidate_headers[0][1].strip()

            if best_header_full == span["text"].strip():
                continue

            span["header"] = normalize_header_key(best_header_full)

    return pii_spans

# ===================== Confidence Assignment =====================

def assign_confidence(pii_spans):
    """
    Add a simple confidence score:
    - 1.0 if has header
    - 0.7 if no header but near PERSON/ORG (same or ±1 line)
    - 0.4 otherwise
    """
    persons_orgs_by_line = {}
    for span in pii_spans:
        if span["pii_type"] in ("PERSON", "ORG"):
            line = span.get("line_idx")
            if line is not None:
                persons_orgs_by_line.setdefault(line, []).append(span)

    for span in pii_spans:
        if "header" in span:
            span["confidence"] = 1.0
            continue

        line = span.get("line_idx")
        nearby_entity = False
        if line is not None:
            for delta in (-1, 0, 1):
                if line + delta in persons_orgs_by_line:
                    nearby_entity = True
                    break

        if nearby_entity:
            span["confidence"] = 0.7
        else:
            span["confidence"] = 0.4

    return pii_spans

# ===================== Owner Selection & Assignment =====================

def is_good_owner_entity(span):
    """Decide if a PERSON/ORG span can be used as a grouping key (owner)."""
    if span["pii_type"] not in GOOD_OWNER_LABELS:
        return False

    txt = span["text"].strip()
    if txt in BAD_ENTITY_KEYS:
        return False

    if not re.search(r"[A-Za-z]", txt):
        return False

    return True


def assign_owner_to_unlabeled_pii(pii_spans):
    """
    Assign PERSON/ORG owners to all non-owner spans:
    - Prefer nearest PERSON within ±max_delta lines.
    - If no PERSON found, fall back to nearest ORG.
    """
    person_owners = []
    org_owners = []

    for span in pii_spans:
        if span["pii_type"] == "PERSON":
            line = span.get("line_idx")
            if line is not None and is_good_owner_entity(span):
                person_owners.append((line, span["text"].strip()))
        elif span["pii_type"] == "ORG":
            line = span.get("line_idx")
            if line is not None and is_good_owner_entity(span):
                org_owners.append((line, span["text"].strip()))

    person_owners.sort(key=lambda x: x[0])
    org_owners.sort(key=lambda x: x[0])

    def find_nearest_owner(line_idx, owners, max_delta=5):
        best_name = None
        best_dist = None
        for owner_line, owner_name in owners:
            delta = line_idx - owner_line
            if abs(delta) > max_delta:
                continue
            if best_dist is None or abs(delta) < best_dist:
                best_dist = abs(delta)
                best_name = owner_name
        return best_name

    for span in pii_spans:
        # skip owner spans themselves
        if span["pii_type"] in ("PERSON", "ORG"):
            continue

        line = span.get("line_idx")
        if line is None:
            continue

        owner_name = find_nearest_owner(line, person_owners)
        if not owner_name:
            owner_name = find_nearest_owner(line, org_owners)

        if owner_name:
            span["owner"] = owner_name

    return pii_spans

# ===================== Clustering by Header or Entity =====================

def cluster_pii_by_header_or_entity(text: str, pii_spans: list):
    """
    Group PII spans by owner (preferred) or header, else NO_HEADER.
    """
    clusters = {}
    no_header_key = "NO_HEADER"

    for span in pii_spans:
        header_key = span.get("header")
        owner_key = span.get("owner")

        if owner_key:
            clusters.setdefault(owner_key, []).append(span)
        elif header_key:
            clusters.setdefault(header_key, []).append(span)
        else:
            clusters.setdefault(no_header_key, []).append(span)

    return clusters

# ===================== Owner-centric View =====================

def build_owner_view(pii_spans):
    """
    Build an owner-centric view:
    {
      "Owner Name or Header": {
        "phones": [...],
        "ssn": [...],
        "addresses": [...],
        "emails": [...],
        "dates": [...],
        "other": [...]
      },
      ...
    }
    """
    owner_view = {}

    for span in pii_spans:
        owner = span.get("owner") or span.get("header")
        if not owner:
            continue

        bucket = owner_view.setdefault(
            owner,
            {
                "phones": [],
                "ssn": [],
                "addresses": [],
                "emails": [],
                "dates": [],
                "other": [],
            },
        )

        t = span["pii_type"]
        txt = span["text"]

        if t == "PHONE":
            bucket["phones"].append(txt)
        elif t == "SSN":
            bucket["ssn"].append(txt)
        elif t in ("GPE", "ZIPCODE", "LOC"):
            bucket["addresses"].append(txt)
        elif t in ("EMAIL", "EMAIL_ADDRESS"):
            bucket["emails"].append(txt)
        elif t == "DATE":
            bucket["dates"].append(txt)
        else:
            bucket["other"].append({"type": t, "text": txt})

    return owner_view

# ===================== Conflict Resolution =====================

def resolve_conflicts(spans):
    """If the same text appears with multiple labels, keep highest-priority one."""
    priority = [
        "SSN",
        "CREDIT_CARD",
        "DLN",
        "EMAIL",
        "PHONE",
        "ZIPCODE",
        "DATE",
        "TIME",
        "CARD_LAST4",
        "AUTH_CODE",
        "PERSON",
        "ORG",
        "GPE",
        "NORP",
        "URL",
        "CURRENCY",
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
    parser.add_argument("--input-dir", default="./data/input")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", default="./data/output")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load config (chunk size)
    try:
        conf = json.loads(Path("app_config.json").read_text(encoding="utf-8"))
        chunk_size = conf.get("chunk_size", args.chunk_size)
    except Exception:
        chunk_size = args.chunk_size or 7000

    # Load spaCy once (important for speed)
    nlp = spacy.load(SPACY_MODEL)

    # Iterate over *all JSON files*
    for json_path in sorted(input_dir.glob("*.json")):
        print(f"\n🔍 Processing: {json_path.name}")

        # Read OCR text
        text = extract_text_from_json(json_path)

        all_pii = []

        # Chunking
        for idx, (chunk, (start, _)) in enumerate(split_text_into_chunks(text, chunk_size)):
            sp_hits = extract_spacy_entities(nlp, chunk)
            regex_hits = extract_regex_pii(chunk)

            merged = []
            for hit in sp_hits + regex_hits:
                new = dict(hit)
                new["start"] += start
                new["end"] += start
                new["chunk"] = idx
                merged.append(new)

            cleaned = resolve_conflicts(merged)
            all_pii.extend(cleaned)

        # Post processing
        all_pii = attach_headers_to_pii(text, all_pii)
        all_pii = assign_confidence(all_pii)
        all_pii = assign_owner_to_unlabeled_pii(all_pii)

        clusters = cluster_pii_by_header_or_entity(text, all_pii)
        owner_view = build_owner_view(all_pii)

        # ------ SAVE EACH FILE WITH ITS OWN NAME ------
        base = json_path.stem  # Example: Vamp0000042680_ocr

        (output_dir / f"{base}_PII.json").write_text(json.dumps(all_pii, indent=2))
        (output_dir / f"{base}_PII_grouped.json").write_text(json.dumps(clusters, indent=2))
        (output_dir / f"{base}_PII_by_owner.json").write_text(json.dumps(owner_view, indent=2))

        print(f"✔ Finished {json_path.name}")

    print("\n🎉 ALL FILES PROCESSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
