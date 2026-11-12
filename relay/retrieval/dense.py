"""
FastEmbed Dense Embedding Engine for The Relay.
Runs locally using ONNX runtime and BAAI/bge-small-en-v1.5.
Zero API calls, zero quota spent, sub-10ms per search.
"""

import os
import numpy as np
from typing import Any, Dict, List, Tuple
from fastembed import TextEmbedding


class DenseSearcher:
    def __init__(self, chunks: List[Dict[str, Any]], model_name: str = "BAAI/bge-small-en-v1.5"):
        self.chunks = chunks
        self.model = TextEmbedding(model_name=model_name)
        self.corpus_texts = [
            f"{c['paper_title']} | Section: {c['section']} | {c['text']}" for c in chunks
        ]
        print(f"Embedding {len(self.corpus_texts)} chunks with {model_name}...")
        # Compute embeddings as float32 numpy array
        embeddings_gen = self.model.embed(self.corpus_texts)
        self.corpus_embeddings = np.array(list(embeddings_gen), dtype=np.float32)
        # Normalize for cosine similarity via dot product
        norms = np.linalg.norm(self.corpus_embeddings, axis=1, keepdims=True)
        self.corpus_embeddings = self.corpus_embeddings / np.maximum(norms, 1e-12)
        print("Embeddings ready.")

    def search(self, query: str, top_k: int = 10, paper_hint: str = None) -> List[Tuple[Dict[str, Any], float]]:
        q_emb_gen = self.model.embed([f"Represent this sentence for searching relevant passages: {query}"])
        q_emb = np.array(list(q_emb_gen)[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = np.dot(self.corpus_embeddings, q_emb)

        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            final_score = float(score)

            if chunk.get("chunk_type") == "reference" and "reference" not in query.lower():
                final_score *= 0.3

            if paper_hint:
                hint_lower = paper_hint.lower()
                if hint_lower in chunk["paper_id"].lower() or hint_lower in chunk["paper_title"].lower():
                    final_score *= 1.25

            scored_chunks.append((chunk, final_score))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

