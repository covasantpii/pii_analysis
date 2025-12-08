"""
OCR JSON Loader - Extract text and metadata from OCR JSON files
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


OCR_JSON_SCHEMA = {
    "pages": [
        {
            "page_num": 1,
            "text": "full page text here..."
        }
    ]
}


def load_ocr_json(path: str | Path) -> Dict[str, Any]:
    """
    Load OCR JSON file and validate structure.
    
    Args:
        path: Path to OCR JSON file
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"OCR JSON file not found: {path}")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def extract_text_from_ocr(ocr_json: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract concatenated text from OCR JSON and track page segments.
    
    Supports multiple OCR JSON formats:
    1. {"pages": [{"page_num": int, "text": str}, ...]}
    2. {"ocr_text": str} (single text field - primary format)
    3. {"ocr_text": str, "page_texts": {str: str}} (Vision API format)
    
    Args:
        ocr_json: Parsed OCR JSON in any supported format
    
    Returns:
        Tuple of:
            - full_text: Concatenated text from all pages (joined with newlines)
            - segments: List of dicts with page metadata and character offsets
    """
    texts: List[str] = []
    segments: List[Dict[str, Any]] = []
    
    # Priority 1: Use "ocr_text" if available (most complete extraction)
    if "ocr_text" in ocr_json:
        full_text = ocr_json.get("ocr_text", "")
        # Try to break into pages using page_texts structure if available
        page_texts = ocr_json.get("page_texts", {})
        
        if page_texts:
            # Use page_texts to create segments
            offset = 0
            for page_num_str in sorted(page_texts.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                page_text = page_texts.get(page_num_str, "") or ""
                start = offset
                end = offset + len(page_text)
                
                # Try to convert page number to int, otherwise use string
                try:
                    page_num = int(page_num_str) + 1  # Convert 0-indexed to 1-indexed
                except (ValueError, TypeError):
                    page_num = page_num_str
                
                segments.append({
                    "page_num": page_num,
                    "start": start,
                    "end": end,
                })
                offset = end + 1
        else:
            # No page_texts, treat entire ocr_text as single page
            segments.append({
                "page_num": 1,
                "start": 0,
                "end": len(full_text),
            })
        
        return full_text, segments
    
    # Priority 2: Use "pages" array structure
    pages = ocr_json.get("pages", [])
    if pages:
        offset = 0
        for i, page in enumerate(pages):
            page_text = page.get("text", "") or ""
            start = offset
            end = start + len(page_text)
            texts.append(page_text)
            segments.append({
                "page_num": page.get("page_num", i + 1),
                "start": start,
                "end": end,
            })
            offset = end + 1
            texts.append("\n")
        
        full_text = "".join(texts)
        return full_text, segments
    
    # If no recognized format found, return empty
    return "", []
