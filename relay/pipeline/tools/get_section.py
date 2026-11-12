"""
Tool 1: get_section(paper, section_or_page)
Returns the verbatim text of an entire section or page from the canonical document store.
Used when retrieved chunks are clipped or need surrounding context.
"""

import json
import os
from typing import Any, Dict, List, Optional

CANONICAL_PATH = "corpus/processed/corpus_canonical.json"


def get_section(paper_id_or_keyword: str, section_or_page: str) -> Dict[str, Any]:
    if not os.path.exists(CANONICAL_PATH):
        return {"error": "Canonical corpus not found."}

    with open(CANONICAL_PATH, "r", encoding="utf-8") as f:
        blocks: List[Dict[str, Any]] = json.load(f)

    paper_needle = paper_id_or_keyword.strip().lower()
    target_needle = section_or_page.strip().lower()

    # Filter to paper
    paper_blocks = [
        b for b in blocks
        if paper_needle in b["paper_id"].lower() or paper_needle in b["paper_title"].lower()
    ]
    if not paper_blocks:
        return {
            "found": False,
            "message": f"No paper matching '{paper_id_or_keyword}' found."
        }

    # Match by page number if numeric
    if target_needle.isdigit():
        target_page = int(target_needle)
        matched = [b for b in paper_blocks if b.get("page") == target_page]
        if matched:
            full_text = "\n\n".join(b["text"] for b in matched)
            return {
                "found": True,
                "paper_id": matched[0]["paper_id"],
                "paper_title": matched[0]["paper_title"],
                "page": target_page,
                "section": matched[0]["section"],
                "text": full_text
            }

    # Match by section title
    matched = [
        b for b in paper_blocks
        if target_needle in b.get("section", "").lower()
    ]
    if matched:
        full_text = "\n\n".join(b["text"] for b in matched)
        pages = sorted(list(set(b.get("page", 0) for b in matched)))
        return {
            "found": True,
            "paper_id": matched[0]["paper_id"],
            "paper_title": matched[0]["paper_title"],
            "section": matched[0]["section"],
            "pages": pages,
            "text": full_text
        }

    return {
        "found": False,
        "message": f"Section or page '{section_or_page}' not found in paper '{paper_id_or_keyword}'."
    }

