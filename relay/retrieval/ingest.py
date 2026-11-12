"""
PDF Ingestion & Structured Canonical Store Builder for The Relay.
Parses the 3 pinned arXiv research papers into structured blocks preserving:
- paper_id
- title
- page_number
- section_title
- chunk_type (header, body, table, caption, reference)
- raw_text
"""

import json
import os
import re
from typing import Any, Dict, List
import pypdf

RAW_PAPERS = [
    {
        "id": "1706.03762v7",
        "title": "Attention Is All You Need",
        "path": "corpus/raw/1706.03762v7_attention_is_all_you_need.pdf"
    },
    {
        "id": "2410.00037v1",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "path": "corpus/raw/2410.00037v1_moshi.pdf"
    },
    {
        "id": "2507.07955v2",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "path": "corpus/raw/2507.07955v2_dynamic_chunking.pdf"
    }
]

SECTION_PATTERN = re.compile(
    r"^([0-9]+(\.[0-9]+)*\s+[A-Z][A-Za-z0-9\s,\-–—]+|Abstract|References|Conclusion|Introduction|Discussion|Related Work|Background)",
    re.MULTILINE
)

TABLE_PATTERN = re.compile(r"^(Table\s+[0-9]+[:\.\s])", re.IGNORECASE)
FIGURE_PATTERN = re.compile(r"^(Figure\s+[0-9]+[:\.\s])", re.IGNORECASE)


def parse_pdf(paper_meta: Dict[str, str]) -> List[Dict[str, Any]]:
    paper_id = paper_meta["id"]
    title = paper_meta["title"]
    pdf_path = paper_meta["path"]

    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Parsing {title} ({paper_id}): {total_pages} pages...")

    blocks = []
    current_section = "Introduction"
    in_references = False

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text() or ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        current_para = []
        for line in lines:
            # Check for Reference section
            if re.search(r"^(References|Bibliography)$", line, re.IGNORECASE):
                in_references = True
                current_section = "References"

            # Check for section header
            sec_match = SECTION_PATTERN.match(line)
            if sec_match and len(line) < 80 and not in_references:
                if current_para:
                    para_text = " ".join(current_para).strip()
                    if para_text:
                        blocks.append({
                            "paper_id": paper_id,
                            "paper_title": title,
                            "page": page_num,
                            "section": current_section,
                            "chunk_type": "body",
                            "text": para_text
                        })
                    current_para = []
                current_section = line
                blocks.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page": page_num,
                    "section": current_section,
                    "chunk_type": "header",
                    "text": line
                })
                continue

            # Check for Table or Figure caption
            if TABLE_PATTERN.match(line):
                if current_para:
                    blocks.append({
                        "paper_id": paper_id,
                        "paper_title": title,
                        "page": page_num,
                        "section": current_section,
                        "chunk_type": "body",
                        "text": " ".join(current_para).strip()
                    })
                    current_para = []
                blocks.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page": page_num,
                    "section": current_section,
                    "chunk_type": "table",
                    "text": line
                })
                continue

            if FIGURE_PATTERN.match(line):
                if current_para:
                    blocks.append({
                        "paper_id": paper_id,
                        "paper_title": title,
                        "page": page_num,
                        "section": current_section,
                        "chunk_type": "body",
                        "text": " ".join(current_para).strip()
                    })
                    current_para = []
                blocks.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page": page_num,
                    "section": current_section,
                    "chunk_type": "caption",
                    "text": line
                })
                continue

            # Accumulate normal body lines into paragraphs
            current_para.append(line)
            # Break paragraph on likely sentence end or paragraph boundary
            if line.endswith((".", ":", ";")) and len(current_para) >= 4:
                para_text = " ".join(current_para).strip()
                blocks.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page": page_num,
                    "section": current_section,
                    "chunk_type": "reference" if in_references else "body",
                    "text": para_text
                })
                current_para = []

        if current_para:
            para_text = " ".join(current_para).strip()
            if para_text:
                blocks.append({
                    "paper_id": paper_id,
                    "paper_title": title,
                    "page": page_num,
                    "section": current_section,
                    "chunk_type": "reference" if in_references else "body",
                    "text": para_text
                })

    print(f"  Extracted {len(blocks)} blocks from {paper_id}")
    return blocks


def main():
    os.makedirs("corpus/processed", exist_ok=True)
    all_blocks = []
    
    for paper in RAW_PAPERS:
        if not os.path.exists(paper["path"]):
            raise FileNotFoundError(f"Missing PDF: {paper['path']}")
        paper_blocks = parse_pdf(paper)
        all_blocks.extend(paper_blocks)

    canonical_path = "corpus/processed/corpus_canonical.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(all_blocks, f, indent=2, ensure_ascii=False)
    
    print(f"\nCanonical corpus written to {canonical_path} ({len(all_blocks)} total blocks).")


if __name__ == "__main__":
    main()

