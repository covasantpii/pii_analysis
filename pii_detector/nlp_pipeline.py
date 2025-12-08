"""
NLP Pipeline - Load spaCy model and analyze text for PII with chunking support
"""

from typing import List, Dict, Any, Tuple
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from . import pii_rules
from config import SPACY_MODEL


# Module-level singleton for spaCy model (lazy-loaded)
_nlp: Language | None = None

# Optimal chunk size for spaCy processing (characters)
OPTIMAL_CHUNK_SIZE = 10000  # 10KB chunks for better memory efficiency


def get_nlp() -> Language:
    """
    Get or initialize the spaCy model (lazy loading).
    Uses module-level caching to avoid reloading the model multiple times.
    
    Returns:
        Initialized spaCy Language model
        
    Raises:
        OSError: If spaCy model is not installed
    """
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL)
        except OSError:
            raise OSError(
                f"spaCy model '{SPACY_MODEL}' not found. "
                f"Install it with: python -m spacy download {SPACY_MODEL.split('_')[0]}_core_web_sm"
            )
    return _nlp
def split_into_chunks(full_text, chunk_size=5000):
    """
    Generator that yields chunk_text and its character index span.
    No list stored → memory safe.
    """
    text_length = len(full_text)
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)

        yield full_text[start:end], (start, end)

        start = end

def analyze_text(text: str, deduplicate: bool = False, merge_names: bool = True, use_chunking: bool = True) -> Dict[str, Any]:
    """
    Run spaCy NER and regex PII detection on text with optional chunking.
    Chunks text into optimal segments to better preserve entity boundaries.
    
    Args:
        text: Text to analyze
        deduplicate: If True, remove overlapping/duplicate PII spans
        merge_names: If True, merge fragmented person names detected separately
        use_chunking: If True, split text into chunks for processing (better for large texts)
        
    Returns:
        Dictionary with analysis results:
        {
            "text_length": int,
            "pii_spans": [
                {
                    "source": "spacy|regex|merged",
                    "label": "PERSON|EMAIL|...",
                    "pii_type": "PERSON|EMAIL|...",
                    "text": "detected text",
                    "start": int,
                    "end": int
                },
                ...
            ]
        }
    """
    nlp = get_nlp()
    all_hits = []
    
    if use_chunking and len(text) > OPTIMAL_CHUNK_SIZE:
        # Process text in chunks with overlap
        chunks = split_into_chunks(text, OPTIMAL_CHUNK_SIZE, overlap=500)
        
        for chunk_text, chunk_start in chunks:
            doc: Doc = nlp(chunk_text)
            
            # Extract entities from this chunk
            entities = pii_rules.extract_spacy_entities(doc)
            regex_hits = pii_rules.extract_regex_pii(chunk_text)
            
            # Adjust offsets to global text positions
            for hit in entities + regex_hits:
                hit["start"] += chunk_start
                hit["end"] += chunk_start
            
            all_hits.extend(entities + regex_hits)
    else:
        # Process entire text at once (for small texts)
        doc: Doc = nlp(text)
        entities = pii_rules.extract_spacy_entities(doc)
        regex_hits = pii_rules.extract_regex_pii(text)
        all_hits = entities + regex_hits
    
    # Sort by start index for consistent ordering
    all_hits = sorted(all_hits, key=lambda x: x["start"])
    
    # Optionally merge fragmented names
    if merge_names:
        all_hits = pii_rules.merge_fragmented_names(text, all_hits)
        all_hits = sorted(all_hits, key=lambda x: x["start"])
    
    # Optionally remove overlaps
    if deduplicate:
        all_hits = pii_rules.deduplicate_spans(all_hits)

    return {
        "text_length": len(text),
        "pii_spans": all_hits,
    }
