"""
Chunking and Indexing Strategies for The Relay.
Implements the 4 retrieval settings from the brief:
- R1: ~200 tokens/chunk, fetch 8 pieces, no rerank
- R2: ~400 tokens/chunk (paragraph-aware, 15% overlap), fetch 5 pieces, no rerank (Default)
- R3: ~400 tokens/chunk, fetch 20 candidates, rerank top 5 (simulated or cross-encoder)
- R4: ~800 tokens/chunk, fetch 3 pieces, no rerank
"""

import json
import os
import re
from typing import Any, Dict, List

CANONICAL_PATH = "corpus/processed/corpus_canonical.json"


def approx_token_count(text: str) -> int:
    # 1 token ~= 4 chars or ~0.75 words. Using whitespace + punctuation splits for deterministic estimate.
    words = re.findall(r"\w+|[^\w\s]", text)
    return max(1, int(len(words) * 1.1))


def create_chunks(target_tokens: int, overlap_percent: float = 0.0) -> List[Dict[str, Any]]:
    if not os.path.exists(CANONICAL_PATH):
        raise FileNotFoundError(f"Missing {CANONICAL_PATH}. Run ingest.py first.")

    with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    chunks = []
    chunk_counter = 0

    # Group by (paper_id, section) to avoid merging across sections
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for block in blocks:
        key = f"{block['paper_id']}|||{block['section']}"
        grouped.setdefault(key, []).append(block)

    overlap_tokens = int(target_tokens * overlap_percent)

    for key, sec_blocks in grouped.items():
        curr_text_parts = []
        curr_tokens = 0
        curr_pages = set()
        paper_id = sec_blocks[0]["paper_id"]
        paper_title = sec_blocks[0]["paper_title"]
        section = sec_blocks[0]["section"]
        is_ref = any(b["chunk_type"] == "reference" for b in sec_blocks)

        for b in sec_blocks:
            b_text = b["text"]
            b_tokens = approx_token_count(b_text)
            curr_pages.add(b["page"])

            # If it's a standalone table or header, treat carefully
            if b["chunk_type"] in ["table", "caption"]:
                curr_text_parts.append(f"[{b['chunk_type'].upper()}: {b_text}]")
            else:
                curr_text_parts.append(b_text)

            curr_tokens += b_tokens

            if curr_tokens >= target_tokens:
                full_text = "\n\n".join(curr_text_parts)
                chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:04d}",
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "section": section,
                    "pages": sorted(list(curr_pages)),
                    "page_str": ", ".join(str(p) for p in sorted(list(curr_pages))),
                    "chunk_type": "reference" if is_ref else "body",
                    "approx_tokens": curr_tokens,
                    "text": full_text
                })
                chunk_counter += 1

                # Handle overlap
                if overlap_tokens > 0 and len(curr_text_parts) > 1:
                    # Keep tail elements
                    tail_parts = []
                    tail_tokens = 0
                    for p in reversed(curr_text_parts):
                        tail_tokens += approx_token_count(p)
                        tail_parts.insert(0, p)
                        if tail_tokens >= overlap_tokens:
                            break
                    curr_text_parts = tail_parts
                    curr_tokens = tail_tokens
                else:
                    curr_text_parts = []
                    curr_tokens = 0
                    curr_pages = set()

        if curr_text_parts:
            full_text = "\n\n".join(curr_text_parts)
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter:04d}",
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section": section,
                "pages": sorted(list(curr_pages)),
                "page_str": ", ".join(str(p) for p in sorted(list(curr_pages))),
                "chunk_type": "reference" if is_ref else "body",
                "approx_tokens": curr_tokens,
                "text": full_text
            })
            chunk_counter += 1

    return chunks


def build_all_settings():
    os.makedirs("corpus/processed", exist_ok=True)
    settings = {
        "R1": {"target_tokens": 200, "overlap": 0.0, "k": 8, "rerank": False},
        "R2": {"target_tokens": 400, "overlap": 0.15, "k": 5, "rerank": False},
        "R3": {"target_tokens": 400, "overlap": 0.15, "k": 5, "rerank": True, "k_candidates": 20},
        "R4": {"target_tokens": 800, "overlap": 0.0, "k": 3, "rerank": False},
    }

    # Build unique chunk sets
    chunks_200 = create_chunks(200, 0.0)
    chunks_400 = create_chunks(400, 0.15)
    chunks_800 = create_chunks(800, 0.0)

    with open("corpus/processed/chunks_r1.json", "w", encoding="utf-8") as f:
        json.dump(chunks_200, f, indent=2, ensure_ascii=False)
    print(f"R1 chunks (200 tok): {len(chunks_200)}")

    with open("corpus/processed/chunks_r2_r3.json", "w", encoding="utf-8") as f:
        json.dump(chunks_400, f, indent=2, ensure_ascii=False)
    print(f"R2/R3 chunks (400 tok): {len(chunks_400)}")

    with open("corpus/processed/chunks_r4.json", "w", encoding="utf-8") as f:
        json.dump(chunks_800, f, indent=2, ensure_ascii=False)
    print(f"R4 chunks (800 tok): {len(chunks_800)}")

    meta_path = "corpus/processed/retrieval_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    print(f"Retrieval settings metadata written to {meta_path}")


if __name__ == "__main__":
    build_all_settings()

