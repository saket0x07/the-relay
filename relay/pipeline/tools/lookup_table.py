"""
Tool 2: lookup_table(paper, table_id)
Extracts and structures tabular data and performance metrics from the papers.
Prevents scrambled table figures and OCR layout damage from producing hallucinations.
"""

import json
import os
import re
from typing import Any, Dict, List

CANONICAL_PATH = "corpus/processed/corpus_canonical.json"


def lookup_table(paper_id_or_keyword: str, table_id: str) -> Dict[str, Any]:
    if not os.path.exists(CANONICAL_PATH):
        return {"error": "Canonical corpus not found."}

    with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
        blocks: List[Dict[str, Any]] = json.load(f)

    paper_needle = paper_id_or_keyword.strip().lower()
    table_needle = table_id.strip().lower()

    paper_blocks = [
        b for b in blocks
        if paper_needle in b["paper_id"].lower() or paper_needle in b["paper_title"].lower()
    ]
    if not paper_blocks:
        return {"found": False, "message": f"Paper '{paper_id_or_keyword}' not found."}

    # Search blocks of type table or blocks containing "Table <id>"
    matched_blocks = []
    for b in paper_blocks:
        t = b["text"].lower()
        if f"table {table_needle}" in t or f"table {table_needle}:" in t or f"table {table_needle}." in t:
            matched_blocks.append(b)

    if not matched_blocks:
        # Check without prefix
        for b in paper_blocks:
            if b["chunk_type"] == "table" and table_needle in b["text"].lower():
                matched_blocks.append(b)

    if matched_blocks:
        combined_text = "\n\n".join(b["text"] for b in matched_blocks)
        pages = sorted(list(set(b["page"] for b in matched_blocks)))
        return {
            "found": True,
            "paper_id": matched_blocks[0]["paper_id"],
            "table_id": table_id,
            "pages": pages,
            "section": matched_blocks[0]["section"],
            "raw_table_content": combined_text
        }

    return {
        "found": False,
        "message": f"Table '{table_id}' not found in paper '{paper_id_or_keyword}'."
    }

