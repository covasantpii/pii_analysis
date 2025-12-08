"""
Text Redaction - Mask PII spans in text
"""

from typing import List, Dict, Any


def redact_text(text: str, spans: List[Dict[str, Any]], mask_char: str = "X") -> str:
    """
    Redact PII spans by replacing characters with mask_char.
    
    Args:
        text: Original text
        spans: List of PII spans with 'start' and 'end' character offsets
        mask_char: Character to use for masking (default: 'X')
        
    Returns:
        Redacted text with PII replaced by mask_char
    """
    chars = list(text)
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        # Ensure we don't go out of bounds
        for i in range(start, min(end, len(chars))):
            chars[i] = mask_char
    return "".join(chars)


def redact_text_with_tags(text: str, spans: List[Dict[str, Any]]) -> str:
    """
    Redact PII spans by replacing with [PII_TYPE] tags instead of masking.
    Useful for reviewing what was detected without seeing actual values.
    
    Args:
        text: Original text
        spans: List of PII spans with 'start', 'end', and 'pii_type'
        
    Returns:
        Text with PII replaced by [PII_TYPE] tags
    """
    # Sort spans in reverse order to maintain correct offsets during replacement
    sorted_spans = sorted(spans, key=lambda x: x["start"], reverse=True)
    
    chars = list(text)
    for span in sorted_spans:
        start = int(span["start"])
        end = int(span["end"])
        pii_type = span.get("pii_type", "PII")
        
        if start < len(chars) and end <= len(chars):
            replacement = f"[{pii_type}]"
            chars[start:end] = list(replacement)
    
    return "".join(chars)
