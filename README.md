# PII Detection using spaCy NER

Detects and redacts Personally Identifiable Information (PII) from OCR-extracted text using spaCy Named Entity Recognition + regex patterns.

## Features
- ✅ **Dual Detection**: spaCy NER (PERSON, ORG, GPE) + regex (EMAIL, PHONE, SSN)
- ✅ **Multi-page OCR**: Handles OCR JSON with character offset tracking
- ✅ **Chunk Processing**: Processes large documents without memory issues
- ✅ **Multiple Redaction**: Masking or PIITYPE tags
- ✅ **Chunk Outputs**: Individual files per chunk

## Installation
