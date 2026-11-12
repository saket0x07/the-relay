"""
Technical Tokenizer & BM25 Search Engine for The Relay.
Preserves technical compound terms, hyphens, and identifiers like:
- d_model, dim_feedforward, rq-vae, BLEU-4, v1.0, 1706.03762v7
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple
from rank_bm25 import BM25Okapi

# Regex preserving technical terms, underscores, periods inside numbers, and hyphens
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[_\-\.][A-Za-z0-9]+)*", re.ASCII)


def technical_tokenize(text: str) -> List[str]:
    text_clean = text.lower()
    return TOKEN_PATTERN.findall(text_clean)


class BM25Searcher:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus_tokens = [technical_tokenize(c["text"] + " " + c["section"] + " " + c["paper_title"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10, paper_hint: str = None) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = technical_tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)

        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            final_score = float(score)

            # Downweight references unless specifically requested
            if chunk.get("chunk_type") == "reference" and "reference" not in query.lower():
                final_score *= 0.15

            # Apply soft boost for paper hint if provided (not hard filter)
            if paper_hint:
                hint_lower = paper_hint.lower()
                if hint_lower in chunk["paper_id"].lower() or hint_lower in chunk["paper_title"].lower():
                    final_score *= 1.35

            scored_chunks.append((chunk, final_score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

