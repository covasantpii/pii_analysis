"""
Chunk-based Output - Generate separate results for each processing chunk
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def split_spans_by_chunk(pii_spans, chunks):
    """
    Map PII spans to specific chunks.
    chunk format: (chunk_text, (start, end))
    """
    chunk_span_map = [[] for _ in range(len(chunks))]

    for span in pii_spans:
        span_start = span["start"]
        span_end = span["end"]

        for chunk_index, (chunk_text, (chunk_start, chunk_end)) in enumerate(chunks):

            # Check overlap condition
            if not (span_end <= chunk_start or span_start >= chunk_end):
                chunk_span_map[chunk_index].append(span)

    return chunk_span_map


def save_chunk_outputs(
    output_dir: Path,
    base_filename: str,
    chunks: List[Tuple[str, int]],
    chunk_spans: List[List[Dict[str, Any]]],
    full_text: str,
    redacted_text: str,
) -> None:
    """
    Save separate output files for each chunk.
    
    Creates:
    - chunk_001.json, chunk_002.json, etc. (PII results per chunk)
    - chunk_001.txt, chunk_002.txt, etc. (Redacted text per chunk)
    
    Args:
        output_dir: Output directory path
        base_filename: Base filename (without extension)
        chunks: List of (chunk_text, chunk_start_offset) tuples
        chunk_spans: List of PII spans per chunk
        full_text: Full original text
        redacted_text: Full redacted text
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for chunk_idx, (chunk_text, chunk_offset) in enumerate(chunks, 1):
        chunk_num = f"{chunk_idx:03d}"
        
        # Extract redacted portion for this chunk
        chunk_end = chunk_offset + len(chunk_text)
        chunk_redacted = redacted_text[chunk_offset:chunk_end]
        
        # Create chunk JSON result
        chunk_result = {
            "chunk_number": chunk_idx,
            "chunk_offset": chunk_offset,
            "chunk_length": len(chunk_text),
            "pii_count": len(chunk_spans[chunk_idx - 1]),
            "pii_spans": chunk_spans[chunk_idx - 1],
        }
        
        # Save chunk JSON
        json_out = output_dir / f"{base_filename}_chunk_{chunk_num}.json"
        json_out.write_text(json.dumps(chunk_result, indent=2), encoding="utf-8")
        
        # Save chunk text
        txt_out = output_dir / f"{base_filename}_chunk_{chunk_num}_redacted.txt"
        txt_out.write_text(chunk_redacted, encoding="utf-8")
        
        print(f"[OK] Chunk {chunk_num}: {len(chunk_spans[chunk_idx - 1])} PII entities "
              f"saved to {json_out.name} and {txt_out.name}")


def save_chunk_summary(
    output_dir: Path,
    base_filename: str,
    chunks: List[Tuple[str, int]],
    chunk_spans: List[List[Dict[str, Any]]],
) -> None:
    """
    Save a summary CSV showing PII counts per chunk.
    
    Args:
        output_dir: Output directory path
        base_filename: Base filename (without extension)
        chunks: List of (chunk_text, chunk_start_offset) tuples
        chunk_spans: List of PII spans per chunk
    """
    output_dir = Path(output_dir)
    
    summary_out = output_dir / f"{base_filename}_chunk_summary.csv"
    
    with open(summary_out, "w", encoding="utf-8") as f:
        f.write("Chunk,Offset,Length,PII_Count\n")
        
        total_pii = 0
        for chunk_idx, (chunk_text, chunk_offset) in enumerate(chunks, 1):
            pii_count = len(chunk_spans[chunk_idx - 1])
            total_pii += pii_count
            f.write(f"{chunk_idx},{chunk_offset},{len(chunk_text)},{pii_count}\n")
        
        f.write(f"\nTotal,,,{total_pii}\n")
    
    print(f"[OK] Chunk summary saved to {summary_out.name}")
