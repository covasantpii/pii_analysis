"""
PII Extraction Rules - Combine spaCy NER and regex pattern matching
"""

import re
from typing import List, Dict, Any
from spacy.tokens import Doc
from config import PII_ENTITY_LABELS, REGEX_PATTERNS


def extract_spacy_entities(doc: Doc) -> List[Dict[str, Any]]:
    """
    Extract PII entities from spaCy Doc using configured entity labels.
    
    Args:
        doc: spaCy Doc object with entities recognized
        
    Returns:
        List of PII hits with format:
        {
            "source": "spacy",
            "label": "PERSON|GPE|ORG|...",
            "pii_type": "PERSON|GPE|ORG|...",
            "text": "detected text",
            "start": int,  # character offset
            "end": int     # character offset
        }
    """
    hits: List[Dict[str, Any]] = []
    for ent in doc.ents:
        if ent.label_ in PII_ENTITY_LABELS:
            hits.append(
                {
                    "source": "spacy",
                    "label": ent.label_,
                    "pii_type": ent.label_,
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )
    return hits


def extract_regex_pii(text: str) -> List[Dict[str, Any]]:
    """
    Extract PII using regex patterns defined in config.REGEX_PATTERNS.
    
    Args:
        text: Full text to search
        
    Returns:
        List of PII hits with format:
        {
            "source": "regex",
            "label": "EMAIL|PHONE|ID_NUMBER|...",
            "pii_type": "EMAIL|PHONE|ID_NUMBER|...",
            "text": "detected text",
            "start": int,  # character offset
            "end": int     # character offset
        }
    """
    hits: List[Dict[str, Any]] = []
    for pii_type, pattern in REGEX_PATTERNS.items():
        try:
            for match in re.finditer(pattern, text):
                # Map regex pattern types to PII types
                if pii_type == "FULLNAME":
                    pii_label = "PERSON"
                else:
                    pii_label = pii_type
                
                hits.append(
                    {
                        "source": "regex",
                        "label": pii_label,
                        "pii_type": pii_label,
                        "text": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )
        except re.error as e:
            print(f"Warning: Invalid regex pattern for {pii_type}: {e}")
    
    return hits


def deduplicate_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate or overlapping PII spans, keeping the first occurrence.
    Useful when both spaCy and regex detect the same PII.
    
    Args:
        spans: List of PII spans (must be sorted by start position)
        
    Returns:
        Deduplicated list of spans
    """
    if not spans:
        return []
    
    deduped = [spans[0]]
    for span in spans[1:]:
        # Check if this span overlaps with the last kept span
        if span["start"] >= deduped[-1]["end"]:
            deduped.append(span)
    
    return deduped


def merge_fragmented_names(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge fragmented PERSON names that spaCy split across lines or whitespace.
    
    Example: If spaCy extracts "Arden" and nearby "Samuel" on separate lines,
    this attempts to merge them into "Arden, Samuel" or "Arden Samuel".
    
    Args:
        text: Original text
        spans: List of PII spans
        
    Returns:
        List with merged names where applicable
    """
    merged_spans = []
    skip_indices = set()
    
    person_spans = [(i, span) for i, span in enumerate(spans) if span["pii_type"] == "PERSON"]
    
    for idx, (i, span1) in enumerate(person_spans[:-1]):
        if i in skip_indices:
            continue
        
        # Look at next PERSON span
        next_idx, span2 = person_spans[idx + 1]
        
        # Check if spans are close together (likely fragmented name)
        gap = span2["start"] - span1["end"]
        if 0 < gap <= 50:  # Within 50 chars, could be same name
            # Get the text between them
            between = text[span1["end"]:span2["start"]]
            
            # If it's mostly whitespace/newlines or punctuation, might be fragmented
            if between.strip() in ["", ",", ",\n", "\n"]:
                # Merge into a single span
                merged_span = {
                    "source": "merged",
                    "label": "PERSON",
                    "pii_type": "PERSON",
                    "text": text[span1["start"]:span2["end"]],
                    "start": span1["start"],
                    "end": span2["end"],
                }
                merged_spans.append(merged_span)
                skip_indices.add(i)
                skip_indices.add(next_idx)
                continue
        
        # If not merged, keep the original
        if i not in skip_indices:
            merged_spans.append(span1)
    
    # Add last span if not skipped
    if person_spans and person_spans[-1][0] not in skip_indices:
        merged_spans.append(person_spans[-1][1])
    
    return merged_spans
