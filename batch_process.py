"""
Complete OCR + Translation + PII Extraction Pipeline
Features: Separate folders for OCR output and PII extraction
Author: AI Assistant
Date: 2025-11-04
"""

import os
import io
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Google Cloud imports
from google.cloud import vision
from google.cloud import dlp_v2
import google.api_core.exceptions

# Presidio imports
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Other imports
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from langdetect import detect

# Optional: spaCy for advanced NER
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    SPACY_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")

LIKELIHOOD_TO_SCORE_MAPPING = {
    "VERY_UNLIKELY": 0.1,
    "UNLIKELY": 0.3,
    "POSSIBLE": 0.5,
    "LIKELY": 0.7,
    "VERY_LIKELY": 0.9
}

SUPPORTED_LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'hi': 'Hindi', 'ar': 'Arabic',
    'pt': 'Portuguese', 'bn': 'Bengali', 'ru': 'Russian', 'ja': 'Japanese',
    'pa': 'Punjabi', 'de': 'German', 'fr': 'French', 'te': 'Telugu',
    'mr': 'Marathi', 'ta': 'Tamil', 'gu': 'Gujarati', 'kn': 'Kannada',
    'ml': 'Malayalam', 'or': 'Odia', 'ur': 'Urdu', 'it': 'Italian',
    'zh-cn': 'Chinese (Simplified)', 'ko': 'Korean', 'th': 'Thai',
    'vi': 'Vietnamese', 'id': 'Indonesian', 'tr': 'Turkish'
}


# ============================================================================
# PII EXTRACTORS
# ============================================================================

class GoogleDlpPiiExtractor:
    """PII extraction using Google Cloud DLP API."""
    
    def __init__(self, project_id: str, min_likelihood: str = "POSSIBLE"):
        self.project_id = project_id
        self.dlp_client = dlp_v2.DlpServiceClient()
        self.parent = f"projects/{project_id}/locations/global"
        self.min_likelihood = dlp_v2.Likelihood[min_likelihood.upper()]
        
        # Comprehensive PII types
        self.info_types = [
            # Identity
            "PERSON_NAME", "FIRST_NAME", "LAST_NAME", "DATE_OF_BIRTH", "AGE",
            
            # Government IDs
            "INDIA_AADHAAR_INDIVIDUAL", "INDIA_PAN_INDIVIDUAL",
            "US_SOCIAL_SECURITY_NUMBER", "US_PASSPORT", "US_DRIVERS_LICENSE_NUMBER",
            "UK_NATIONAL_INSURANCE_NUMBER", "CANADA_SOCIAL_INSURANCE_NUMBER",
            
            # Financial
            "CREDIT_CARD_NUMBER", "IBAN_CODE", "SWIFT_CODE",
            
            # Contact
            "EMAIL_ADDRESS", "PHONE_NUMBER", "STREET_ADDRESS", "LOCATION",
            
            # Medical
            "MEDICAL_RECORD_NUMBER",
            
            # Other
            "DATE", "TIME"
        ]
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract PII using DLP API."""
        if not text or not text.strip():
            return {
                'findings': [],
                'pii_count': 0,
                'pii_by_type': {},
                'confidence': 0.0,
                'method': 'dlp'
            }
        
        try:
            inspect_config = {
                "info_types": [{"name": it} for it in self.info_types],
                "min_likelihood": self.min_likelihood,
                "include_quote": True,
                "limits": {"max_findings_per_request": 0}
            }
            
            request = {
                "parent": self.parent,
                "inspect_config": inspect_config,
                "item": {"value": text}
            }
            
            response = self.dlp_client.inspect_content(request=request)
            
            findings = []
            pii_by_type = {}
            total_confidence = 0.0
            
            for finding in response.result.findings:
                likelihood = dlp_v2.Likelihood(finding.likelihood).name
                confidence = LIKELIHOOD_TO_SCORE_MAPPING.get(likelihood, 0.5)
                total_confidence += confidence
                
                info_type = finding.info_type.name
                pii_by_type[info_type] = pii_by_type.get(info_type, 0) + 1
                
                findings.append({
                    'text': finding.quote,
                    'type': info_type,
                    'likelihood': likelihood,
                    'confidence': confidence,
                    'start': finding.location.codepoint_range.start,
                    'end': finding.location.codepoint_range.end
                })
            
            avg_confidence = (total_confidence / len(findings)) if findings else 0.0
            
            return {
                'findings': findings,
                'pii_count': len(findings),
                'pii_by_type': pii_by_type,
                'confidence': round(avg_confidence, 2),
                'method': 'dlp'
            }
            
        except google.api_core.exceptions.PermissionDenied:
            return {
                'findings': [],
                'pii_count': 0,
                'pii_by_type': {},
                'confidence': 0.0,
                'method': 'dlp',
                'error': 'DLP API not enabled. Enable at: https://console.cloud.google.com/apis/library/dlp.googleapis.com'
            }
        except Exception as e:
            return {
                'findings': [],
                'pii_count': 0,
                'pii_by_type': {},
                'confidence': 0.0,
                'method': 'dlp',
                'error': str(e)
            }


class PresidioPiiExtractor:
    """Microsoft Presidio PII extraction."""
    
    def __init__(self):
        """Initialize Presidio Analyzer with spaCy NLP engine."""
        try:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            }
            
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            
            self.entity_types = [
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                "IBAN_CODE", "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
                "LOCATION", "DATE_TIME", "NRP", "MEDICAL_LICENSE",
                "URL", "IP_ADDRESS", "CRYPTO",
                "IN_AADHAAR", "IN_PAN", "IN_PASSPORT", "IN_VEHICLE_REGISTRATION"
            ]
            
            self.available = True
            
        except Exception as e:
            print(f"⚠️ Presidio initialization failed: {e}")
            self.available = False
    
    def extract(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Extract PII using Presidio Analyzer."""
        if not self.available or not text or not text.strip():
            return {
                'findings': [],
                'pii_count': 0,
                'pii_by_type': {},
                'confidence': 0.0,
                'method': 'presidio',
                'error': 'Presidio not available' if not self.available else None
            }
        
        try:
            results = self.analyzer.analyze(
                text=text,
                language=language,
                entities=self.entity_types,
                return_decision_process=False,
                score_threshold=0.3
            )
            
            findings = []
            pii_by_type = {}
            total_confidence = 0.0
            
            for result in results:
                entity_text = text[result.start:result.end]
                entity_type = self._map_entity_type(result.entity_type)
                confidence = result.score
                total_confidence += confidence
                
                pii_by_type[entity_type] = pii_by_type.get(entity_type, 0) + 1
                
                findings.append({
                    'text': entity_text,
                    'type': entity_type,
                    'confidence': round(confidence, 2),
                    'start': result.start,
                    'end': result.end
                })
            
            avg_confidence = (total_confidence / len(findings)) if findings else 0.0
            
            return {
                'findings': findings,
                'pii_count': len(findings),
                'pii_by_type': pii_by_type,
                'confidence': round(avg_confidence, 2),
                'method': 'presidio'
            }
            
        except Exception as e:
            return {
                'findings': [],
                'pii_count': 0,
                'pii_by_type': {},
                'confidence': 0.0,
                'method': 'presidio',
                'error': str(e)
            }
    
    def _map_entity_type(self, presidio_type: str) -> str:
        """Map Presidio entity types to standardized types."""
        mapping = {
            'PERSON': 'PERSON_NAME',
            'EMAIL_ADDRESS': 'EMAIL_ADDRESS',
            'PHONE_NUMBER': 'PHONE_NUMBER',
            'CREDIT_CARD': 'CREDIT_CARD_NUMBER',
            'US_SSN': 'US_SOCIAL_SECURITY_NUMBER',
            'US_PASSPORT': 'US_PASSPORT',
            'US_DRIVER_LICENSE': 'US_DRIVERS_LICENSE_NUMBER',
            'LOCATION': 'LOCATION',
            'DATE_TIME': 'DATE',
            'MEDICAL_LICENSE': 'MEDICAL_RECORD_NUMBER',
            'IN_AADHAAR': 'INDIA_AADHAAR_INDIVIDUAL',
            'IN_PAN': 'INDIA_PAN_INDIVIDUAL',
            'IN_PASSPORT': 'INDIA_PASSPORT',
            'IN_VEHICLE_REGISTRATION': 'INDIA_VEHICLE_REGISTRATION'
        }
        return mapping.get(presidio_type, presidio_type)


class SpacyPiiExtractor:
    """Named Entity Recognition using spaCy."""
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy NER."""
        if not SPACY_AVAILABLE or not text:
            return None
        
        try:
            doc = nlp(text)
            
            entities = {
                'PERSON': [], 'ORG': [], 'GPE': [], 'LOC': [],
                'DATE': [], 'TIME': [], 'MONEY': []
            }
            
            for ent in doc.ents:
                if ent.label_ in entities and ent.text.strip():
                    if ent.text.strip() not in entities[ent.label_]:
                        entities[ent.label_].append(ent.text.strip())
            
            entities = {k: v for k, v in entities.items() if v}
            
            return {
                'entities': entities,
                'entity_count': sum(len(v) for v in entities.values()),
                'method': 'spacy'
            }
        except Exception as e:
            return None


# ============================================================================
# OCR & TRANSLATION
# ============================================================================

def extract_text_vision_api(image_path: str) -> Dict[str, Any]:
    """Extract text using Google Vision API."""
    try:
        client = vision.ImageAnnotatorClient()
        
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        response = client.document_text_detection(
            image=image,
            image_context={
                "language_hints": [
                    "en", "hi", "es", "fr", "de", "it", "pt", "ru", "ar", "zh",
                    "ja", "ko", "bn", "pa", "te", "mr", "ta", "gu", "kn", "ml",
                    "or", "ur", "th", "vi", "id", "tr"
                ]
            }
        )
        
        if response.error.message:
            return {
                'success': False,
                'error': response.error.message,
                'text': '',
                'confidence': 0.0
            }
        
        if response.full_text_annotation:
            text = response.full_text_annotation.text
            
            pages = response.full_text_annotation.pages
            if pages:
                total_confidence = sum(page.confidence for page in pages)
                avg_confidence = total_confidence / len(pages)
            else:
                avg_confidence = 0.0
            
            return {
                'success': True,
                'text': text,
                'confidence': round(avg_confidence, 3),
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        else:
            return {
                'success': False,
                'error': 'No text detected',
                'text': '',
                'confidence': 0.0
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text': '',
            'confidence': 0.0
        }


def detect_language(text: str) -> str:
    """Detect language of text."""
    try:
        if not text or len(text.strip()) < 10:
            return 'unknown'
        return detect(text.strip())
    except:
        return 'unknown'


def translate_to_english(text: str) -> Dict[str, Any]:
    """Translate text to English if needed."""
    if not text or len(text.strip()) < 3:
        return {
            'original_text': text,
            'translated_text': text,
            'source_language': 'unknown',
            'source_language_name': 'Unknown',
            'translation_performed': False
        }
    
    try:
        detected_lang = detect_language(text)
        
        if detected_lang == 'en':
            return {
                'original_text': text,
                'translated_text': text,
                'source_language': 'en',
                'source_language_name': 'English',
                'translation_performed': False
            }
        
        max_chunk = 4500
        if len(text) > max_chunk:
            chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
            translated_chunks = []
            
            for chunk in chunks:
                try:
                    trans = GoogleTranslator(source='auto', target='en').translate(chunk)
                    translated_chunks.append(trans)
                except:
                    translated_chunks.append(chunk)
            
            translated_text = ' '.join(translated_chunks)
        else:
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        
        language_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang.upper())
        
        return {
            'original_text': text,
            'translated_text': translated_text,
            'source_language': detected_lang,
            'source_language_name': language_name,
            'translation_performed': True
        }
        
    except Exception as e:
        return {
            'original_text': text,
            'translated_text': text,
            'source_language': 'error',
            'source_language_name': 'Error',
            'translation_performed': False,
            'error': str(e)
        }


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

def process_single_image(
    image_path: str,
    dlp_extractor,
    presidio_extractor,
    spacy_extractor,
    output_dir: str
) -> Dict[str, Any]:
    """Complete processing pipeline for a single image."""
    image_name = Path(image_path).name
    stem = Path(image_path).stem
    
    # Step 1: OCR with Vision API
    print(f"  [1/4] OCR...", end='', flush=True)
    ocr_result = extract_text_vision_api(image_path)
    
    if not ocr_result['success']:
        print(f" ✗")
        return {
            'success': False,
            'image_name': image_name,
            'image_path': str(image_path),
            'error': ocr_result['error'],
            'processed_at': datetime.now().isoformat()
        }
    
    original_text = ocr_result['text']
    print(f" ✓ (conf: {ocr_result['confidence']:.2f})")
    
    # Save parsed OCR output separately
    ocr_output_dir = Path(output_dir) / '1_parsed_ocr_output'
    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    
    ocr_json_path = ocr_output_dir / f"{stem}_ocr.json"
    with open(ocr_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image_name': image_name,
            'image_path': str(image_path),
            'ocr_text': original_text,
            'confidence': ocr_result['confidence'],
            'word_count': ocr_result['word_count'],
            'char_count': ocr_result['char_count'],
            'processed_at': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    # Also save as plain text
    ocr_txt_path = ocr_output_dir / f"{stem}_ocr.txt"
    with open(ocr_txt_path, 'w', encoding='utf-8') as f:
        f.write(original_text)
    
    # Step 2: Translation
    print(f"  [2/4] Translation...", end='', flush=True)
    translation_result = translate_to_english(original_text)
    text_for_pii = translation_result['translated_text']
    lang = translation_result['source_language_name']
    print(f" ✓ ({lang})")
    
    # Step 3: PII Extraction
    print(f"  [3/4] PII Extraction...", end='', flush=True)
    
    if dlp_extractor:
        pii_dlp = dlp_extractor.extract(text_for_pii)
        if pii_dlp.get('error') or pii_dlp['pii_count'] == 0:
            pii_presidio = presidio_extractor.extract(text_for_pii)
            pii_result = pii_presidio
        else:
            pii_result = pii_dlp
    else:
        pii_presidio = presidio_extractor.extract(text_for_pii)
        pii_result = pii_presidio
    
    print(f" ✓ ({pii_result['pii_count']} via {pii_result['method']})")
    
    # Step 4: spaCy NER
    print(f"  [4/4] spaCy NER...", end='', flush=True)
    pii_spacy = None
    if spacy_extractor:
        pii_spacy = spacy_extractor.extract(text_for_pii)
        if pii_spacy:
            print(f" ✓ ({pii_spacy['entity_count']} entities)")
        else:
            print(f" ✗")
    else:
        print(f" (disabled)")
    
    # Build complete result
    result = {
        'success': True,
        'image_name': image_name,
        'image_path': str(image_path),
        'ocr': {
            'original_text': original_text,
            'confidence': ocr_result['confidence'],
            'word_count': ocr_result['word_count'],
            'char_count': ocr_result['char_count']
        },
        'translation': translation_result,
        'pii': pii_result,
        'pii_spacy': pii_spacy,
        'processed_at': datetime.now().isoformat()
    }
    
    # Save extracted PII output separately
    pii_output_dir = Path(output_dir) / '2_extracted_pii_output'
    pii_output_dir.mkdir(parents=True, exist_ok=True)
    
    pii_json_path = pii_output_dir / f"{stem}_pii.json"
    with open(pii_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'image_name': image_name,
            'translation': translation_result,
            'pii_extraction': pii_result,
            'spacy_entities': pii_spacy,
            'processed_at': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    # Save complete combined result
    combined_output_dir = Path(output_dir) / '3_complete_results'
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_json_path = combined_output_dir / f"{stem}_complete.json"
    with open(combined_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def process_directory(search_path: str, output_dir: str):
    """Process all images in a directory."""
    print(f"\n{'='*70}")
    print(f"VISION API + PII EXTRACTION (DLP + PRESIDIO)")
    print(f"Parsed OCR & Extracted PII saved separately")
    print(f"{'='*70}\n")
    
    # Initialize extractors
    dlp_extractor = None
    if GCP_PROJECT_ID:
        try:
            dlp_extractor = GoogleDlpPiiExtractor(GCP_PROJECT_ID, min_likelihood="POSSIBLE")
            print(f"✓ Google DLP API enabled (Project: {GCP_PROJECT_ID})")
        except Exception as e:
            print(f"⚠️ Google DLP API disabled: {e}")
    else:
        print(f"⚠️ GCP_PROJECT_ID not set - DLP disabled")
    
    presidio_extractor = PresidioPiiExtractor()
    if presidio_extractor.available:
        print(f"✓ Microsoft Presidio enabled (NLP-based PII detection)")
    else:
        print(f"❌ Microsoft Presidio initialization failed")
        return
    
    spacy_extractor = SpacyPiiExtractor() if SPACY_AVAILABLE else None
    if SPACY_AVAILABLE:
        print(f"✓ spaCy NER enabled")
    else:
        print(f"⚠️ spaCy not available")
    
    print()
    
    # Find images
    search_dir = Path(search_path)
    image_files = []
    
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        image_files.extend(list(search_dir.rglob(f"*{ext}")))
        image_files.extend(list(search_dir.rglob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"❌ No images found in {search_path}")
        return
    
    print(f"✓ Found {len(image_files)} images\n")
    
    # Process each image
    results = []
    success_count = 0
    total_pii = 0
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        
        result = process_single_image(
            str(img_path),
            dlp_extractor,
            presidio_extractor,
            spacy_extractor,
            output_dir
        )
        
        results.append(result)
        
        if result['success']:
            success_count += 1
            total_pii += result['pii']['pii_count']
    
    # Save summary CSV
    save_summary_csv(results, output_dir)
    
    # Print final summary
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total images: {len(image_files)}")
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {len(image_files) - success_count}")
    print(f"🔍 Total PII found: {total_pii}")
    print(f"\n📁 Output Structure:")
    print(f"   ├── 1_parsed_ocr_output/     (OCR text from Vision API)")
    print(f"   ├── 2_extracted_pii_output/  (PII findings)")
    print(f"   ├── 3_complete_results/      (Combined results)")
    print(f"   └── summary.csv              (Overview table)")
    print(f"{'='*70}\n")


def save_summary_csv(results: List[Dict], output_dir: str):
    """Save summary CSV with all results."""
    csv_path = Path(output_dir) / 'summary.csv'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            'image_name', 'success', 'ocr_confidence', 'source_language',
            'word_count', 'translation_performed', 'pii_count', 'pii_method',
            'pii_types', 'pii_confidence', 'spacy_entities', 'error'
        ])
        
        for result in results:
            if result['success']:
                spacy_count = result['pii_spacy']['entity_count'] if result.get('pii_spacy') else 0
                
                writer.writerow([
                    result['image_name'],
                    result['success'],
                    result['ocr']['confidence'],
                    result['translation']['source_language_name'],
                    result['ocr']['word_count'],
                    result['translation']['translation_performed'],
                    result['pii']['pii_count'],
                    result['pii']['method'],
                    ', '.join(result['pii']['pii_by_type'].keys()),
                    result['pii']['confidence'],
                    spacy_count,
                    ''
                ])
            else:
                writer.writerow([
                    result['image_name'],
                    False, 0.0, 'error', 0, False, 0, 'none', '', 0.0, 0,
                    result.get('error', 'Unknown error')
                ])
    
    print(f"\n✓ Summary CSV saved: {csv_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    SETUP INSTRUCTIONS:
    
    1. Install dependencies:
       pip install google-cloud-vision google-cloud-dlp deep-translator langdetect python-dotenv spacy
       pip install presidio-analyzer presidio-anonymizer
       python -m spacy download en_core_web_sm
    
    2. Create .env file:
       GOOGLE_APPLICATION_CREDENTIALS=gcp_key.json
       GCP_PROJECT_ID=your-project-id
    
    3. Enable DLP API:
       https://console.cloud.google.com/apis/library/dlp.googleapis.com
    
    4. Run:
       python this_file.py
    """
    
    # Configuration
    INPUT_DIR = './visionapi_dataset'
    OUTPUT_DIR = './separated_output'
    
    # Run processing
    process_directory(INPUT_DIR, OUTPUT_DIR)
