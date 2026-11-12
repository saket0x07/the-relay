"""
Reciprocal Rank Fusion (RRF) and Hybrid Search Engine for The Relay.
Merges BM25 lexical rank with Dense embedding rank using formula:
RRF_Score(d) = 1 / (k + rank_bm25) + 1 / (k + rank_dense), where k=60.
"""

from typing import Any, Dict, List, Optional
from relay.retrieval.bm25 import BM25Searcher
from relay.retrieval.dense import DenseSearcher


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    dense_results: List[Dict[str, Any]],
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["chunk_id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

    for rank, doc in enumerate(dense_results):
        doc_id = doc["chunk_id"]
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Sort docs by descending fused score
    sorted_doc_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    
    fused_docs = []
    for d_id in sorted_doc_ids:
        doc_copy = dict(doc_map[d_id])
        doc_copy["rrf_score"] = scores[d_id]
        fused_docs.append(doc_copy)

    return fused_docs


class HybridRetriever:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.bm25 = BM25Searcher(chunks)
        self.dense = DenseSearcher(chunks)

    def search(
        self,
        query: str,
        k_final: int = 5,
        k_candidates: int = 20,
        paper_hint: Optional[str] = None,
        rerank: bool = False
    ) -> List[Dict[str, Any]]:
        # Lexical search
        bm25_hits = self.bm25.search(query, top_k=k_candidates, paper_hint=paper_hint)
        bm25_docs = [hit[0] for hit in bm25_hits]

        # Dense search
        dense_hits = self.dense.search(query, top_k=k_candidates, paper_hint=paper_hint)
        dense_docs = [hit[0] for hit in dense_hits]

        # RRF Fusion
        fused = reciprocal_rank_fusion(bm25_docs, dense_docs, rrf_k=60)

        # Optional simple rerank (e.g. for R3): score by exact keyword coverage in fused candidates
        if rerank and len(fused) > k_final:
            q_terms = set(query.lower().split())
            for doc in fused:
                doc_words = set(doc["text"].lower().split())
                overlap = len(q_terms.intersection(doc_words)) / max(1, len(q_terms))
                doc["rrf_score"] += (overlap * 0.02)
            fused.sort(key=lambda d: d["rrf_score"], reverse=True)

        return fused[:k_final]

