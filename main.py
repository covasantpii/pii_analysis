"""
Main entry point for PII Detection and Redaction - FIXED VERSION
"""

import json
import argparse
from pathlib import Path

from config import DEFAULT_OCR_JSON, OUTPUT_DIR
from pii_detector.ocr_loader import load_ocr_json, extract_text_from_ocr
from pii_detector.nlp_pipeline import analyze_text, split_into_chunks
from pii_detector.redact import redact_text, redact_text_with_tags


def main():
    parser = argparse.ArgumentParser(
        description="PII Detection and Redaction using chunk-based spaCy processing",
    )
    parser.add_argument("--input", type=str, default=DEFAULT_OCR_JSON)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--tag-redaction", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--chunk-output", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # STEP-1 Load OCR JSON
    print(f"\nLoading OCR JSON from: {input_path}")
    ocr_json = load_ocr_json(input_path)

    # STEP-2 Extract full text
    print("Extracting text from OCR...")
    full_text, segments = extract_text_from_ocr(ocr_json)
    print(f"  Total text length: {len(full_text)} characters")
    print(f"  Number of pages: {len(segments)}")

    # STEP-3 Chunk-wise processing
    print("\n🔍 Processing chunks using spaCy …")

    pii_spans = []
    chunk_index = 0

    for chunk_text, (start_idx, end_idx) in split_into_chunks(full_text):
        print(f" → Chunk {chunk_index+1} (chars {start_idx}-{end_idx})")
        # save chunk_text to .txt for debugging


        # ✅ FIXED: Only pass supported arguments
        chunk_result = analyze_text(chunk_text, deduplicate=args.deduplicate)

        # **DE-DUPE MANUALLY** to prevent merged spans
        if not args.deduplicate:
            # Keep ALL spans - no merging/deduplication
            pass
        else:
            # Simple overlap removal (keeps first occurrence)
            chunk_result["pii_spans"] = remove_overlapping_spans(chunk_result["pii_spans"])

        # offset correction
        for span in chunk_result["pii_spans"]:
            span["start"] += start_idx
            span["end"] += start_idx
            span["chunk"] = chunk_index
            pii_spans.append(span)

        if args.verbose:
            for span in chunk_result["pii_spans"]:
                print(f"   [{span['pii_type']}] '{span['text']}' @ {span['start']}-{span['end']}")

        chunk_index += 1

    # STEP-4 Redaction
    print("\nRedacting full document…")
    if args.tag_redaction:
        redacted = redact_text_with_tags(full_text, pii_spans)
    else:
        redacted = redact_text(full_text, pii_spans)

    # STEP-5 Save global results
    print("\nSaving outputs…")

    result = {
        "input_file": str(input_path),
        "text_length": len(full_text),
        "pii_count": len(pii_spans),
        "pii_spans": pii_spans,
        "segment_count": len(segments),
    }

    json_out = out_dir / (input_path.stem + "_pii.json")
    redacted_out = out_dir / (input_path.stem + "_redacted.txt")

    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    redacted_out.write_text(redacted, encoding="utf-8")

    print(f" ✔ Global PII JSON saved to: {json_out}")
    print(f" ✔ Global redacted text saved to: {redacted_out}")

    # STEP-6 Chunk Output Files - SEPARATE PII
    if args.chunk_output:
        print("\n📁 Generating chunk-based individual outputs…")

        chunks_list = []
        current_pos = 0
        chunk_size = 1000
        
        while current_pos < len(full_text):
            chunk_end = min(current_pos + chunk_size, len(full_text))
            chunk_text = full_text[current_pos:chunk_end]
            chunks_list.append({
                "chunk_text": chunk_text,
                "start": current_pos,
                "end": chunk_end,
                "index": len(chunks_list)
            })
            current_pos = chunk_end

        # Group PII spans by chunk
        chunk_spans = {i: [] for i in range(len(chunks_list))}
        for span in pii_spans:
            chunk_idx = span.get('chunk', 0)
            if 0 <= chunk_idx < len(chunk_spans):
                chunk_spans[chunk_idx].append(span)

        # Save individual chunk files
        for chunk_idx, chunk_data in enumerate(chunks_list):
            chunk_pii = sorted(chunk_spans.get(chunk_idx, []), key=lambda x: x['start'])
            
            # Redact this chunk
            chunk_redacted = redact_text(chunk_data["chunk_text"], chunk_pii)
            
            # Chunk JSON output
            chunk_result = {
                "chunk_index": chunk_data["index"],
                "text_length": len(chunk_data["chunk_text"]),
                "pii_count": len(chunk_pii),
                "pii_spans": chunk_pii,
                "original_text": chunk_data["chunk_text"],
                "redacted_text": chunk_redacted
            }
            
            chunk_filename = f"chunk_{chunk_idx+1:02d}.json"
            chunk_path = out_dir / chunk_filename
            chunk_path.write_text(json.dumps(chunk_result, indent=2), encoding="utf-8")
            
            print(f"  ✔ {chunk_filename} ({len(chunk_pii)} PII items)")

        # Chunk summary
        summary = {
            "total_chunks": len(chunks_list),
            "chunks_with_pii": sum(1 for spans in chunk_spans.values() if spans),
            "chunk_pii_distribution": {i: len(spans) for i, spans in chunk_spans.items()}
        }
        summary_path = out_dir / "chunk_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f" ✔ Chunk summary: {summary_path}")

    # STEP-7 Summary
    print("\n=== Summary ===")
    print(f" Total detected PII: {len(pii_spans)}")

    pii_by_type = {}
    for span in pii_spans:
        pii_type = span["pii_type"]
        pii_by_type[pii_type] = pii_by_type.get(pii_type, 0) + 1

    for pii_type, count in sorted(pii_by_type.items()):
        print(f"  {pii_type}: {count}")


def remove_overlapping_spans(spans):
    """Keep first occurrence of overlapping spans - NO MERGING"""
    if not spans:
        return []
    
    # Sort by start position
    sorted_spans = sorted(spans, key=lambda x: x['start'])
    kept_spans = []
    
    for span in sorted_spans:
        # Check if overlaps with any kept span
        overlaps = False
        for kept in kept_spans:
            if (span['start'] < kept['end'] and span['end'] > kept['start']):
                overlaps = True
                break
        
        if not overlaps:
            kept_spans.append(span)
    
    return kept_spans


if __name__ == "__main__":
    main()
