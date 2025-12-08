"""
Unit tests for PII detection system
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pii_detector.ocr_loader import load_ocr_json, extract_text_from_ocr
from pii_detector.nlp_pipeline import analyze_text, get_nlp
from pii_detector.redact import redact_text, redact_text_with_tags
from pii_detector.pii_rules import deduplicate_spans


class TestOCRLoader(unittest.TestCase):
    """Test OCR JSON loading and text extraction"""
    
    def setUp(self):
        self.ocr_data = {
            "pages": [
                {"page_num": 1, "text": "John works at Acme Corp"},
                {"page_num": 2, "text": "Phone: 555-1234"},
            ]
        }
    
    def test_extract_text_from_ocr(self):
        """Test extracting concatenated text and segments"""
        full_text, segments = extract_text_from_ocr(self.ocr_data)
        
        # Verify text is concatenated with newlines
        self.assertIn("John works at Acme Corp", full_text)
        self.assertIn("Phone: 555-1234", full_text)
        
        # Verify segments
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["page_num"], 1)
        self.assertEqual(segments[1]["page_num"], 2)
    
    def test_extract_text_with_empty_page(self):
        """Test handling of empty or missing page text"""
        ocr_data = {
            "pages": [
                {"page_num": 1, "text": "Page 1"},
                {"page_num": 2, "text": ""},  # Empty page
                {"page_num": 3, "text": "Page 3"},
            ]
        }
        full_text, segments = extract_text_from_ocr(ocr_data)
        self.assertEqual(len(segments), 3)
        self.assertIn("Page 1", full_text)
        self.assertIn("Page 3", full_text)


class TestPIIDetection(unittest.TestCase):
    """Test PII detection using spaCy and regex"""
    
    def test_analyze_text_with_person_name(self):
        """Test detection of person names"""
        text = "My name is John Smith and I live in New York"
        result = analyze_text(text)
        
        self.assertIsNotNone(result["pii_spans"])
        # Should detect person names via spaCy NER
        pii_types = [span["pii_type"] for span in result["pii_spans"]]
        self.assertTrue(any(t in ["PERSON", "GPE"] for t in pii_types))
    
    def test_analyze_text_with_email(self):
        """Test detection of email addresses"""
        text = "Contact me at john.doe@example.com for more info"
        result = analyze_text(text)
        
        pii_types = [span["pii_type"] for span in result["pii_spans"]]
        self.assertIn("EMAIL", pii_types)
    
    def test_analyze_text_with_phone(self):
        """Test detection of phone numbers"""
        text = "Call me at 555-123-4567 or +1 (555) 987-6543"
        result = analyze_text(text)
        
        pii_types = [span["pii_type"] for span in result["pii_spans"]]
        self.assertIn("PHONE", pii_types)
    
    def test_analyze_text_with_ssn(self):
        """Test detection of SSN"""
        text = "My SSN is 123-45-6789"
        result = analyze_text(text)
        
        pii_types = [span["pii_type"] for span in result["pii_spans"]]
        self.assertIn("SSN", pii_types)
    
    def test_analyze_text_sorted_by_offset(self):
        """Test that PII spans are sorted by start offset"""
        text = "Email john@example.com or call 555-1234"
        result = analyze_text(text)
        
        spans = result["pii_spans"]
        if len(spans) > 1:
            for i in range(len(spans) - 1):
                self.assertLessEqual(spans[i]["start"], spans[i+1]["start"])


class TestRedaction(unittest.TestCase):
    """Test text redaction functions"""
    
    def test_redact_text_with_masking(self):
        """Test character masking redaction"""
        text = "John Smith works here"
        spans = [
            {"start": 0, "end": 4},    # John
            {"start": 5, "end": 10},   # Smith
        ]
        
        redacted = redact_text(text, spans, mask_char="X")
        self.assertEqual(redacted[:4], "XXXX")
        self.assertEqual(redacted[5:10], "XXXXX")
        self.assertEqual(redacted[10:], " works here")
    
    def test_redact_text_with_tags(self):
        """Test tag-based redaction"""
        text = "John Smith works here"
        spans = [
            {"start": 0, "end": 4, "pii_type": "PERSON"},
        ]
        
        redacted = redact_text_with_tags(text, spans)
        self.assertIn("[PERSON]", redacted)
        self.assertNotIn("John", redacted)
    
    def test_redact_overlapping_spans(self):
        """Test redaction with overlapping spans"""
        text = "test text"
        spans = [
            {"start": 0, "end": 5},
            {"start": 3, "end": 8},
        ]
        
        redacted = redact_text(text, spans)
        # Should mask both spans (overlapping)
        self.assertEqual(redacted, "XXXXXXXXX")


class TestDeduplication(unittest.TestCase):
    """Test PII span deduplication"""
    
    def test_deduplicate_overlapping_spans(self):
        """Test removing overlapping spans"""
        spans = [
            {"start": 0, "end": 5, "pii_type": "PERSON"},
            {"start": 3, "end": 8, "pii_type": "EMAIL"},  # Overlaps
            {"start": 10, "end": 15, "pii_type": "PHONE"},
        ]
        
        deduped = deduplicate_spans(spans)
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["start"], 0)
        self.assertEqual(deduped[1]["start"], 10)
    
    def test_deduplicate_empty_list(self):
        """Test deduplication with empty list"""
        deduped = deduplicate_spans([])
        self.assertEqual(deduped, [])


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline"""
    
    def test_full_pipeline(self):
        """Test complete PII detection pipeline"""
        ocr_data = {
            "pages": [
                {
                    "page_num": 1,
                    "text": "Customer: John Smith\nEmail: john@example.com\nPhone: 555-1234"
                }
            ]
        }
        
        full_text, segments = extract_text_from_ocr(ocr_data)
        result = analyze_text(full_text)
        spans = result["pii_spans"]
        
        # Should find multiple PII items
        self.assertGreater(len(spans), 0)
        
        # Redact text
        redacted = redact_text(full_text, spans)
        self.assertLess(len(redacted.replace("X", "")), len(full_text))


if __name__ == "__main__":
    unittest.main()
