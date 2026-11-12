"""
Precompute and serialize dense embeddings to disk so startup is instantaneous.
"""

import json
import os
import numpy as np
from fastembed import TextEmbedding

def cache_embeddings():
    os.makedirs("corpus/processed/embeddings", exist_ok=True)
    settings = [
        ("r1", "corpus/processed/chunks_r1.json"),
        ("r2_r3", "corpus/processed/chunks_r2_r3.json"),
        ("r4", "corpus/processed/chunks_r4.json")
    ]
    
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    for tag, chunk_path in settings:
        out_path = f"corpus/processed/embeddings/dense_{tag}.npy"
        if os.path.exists(out_path):
            print(f"Embedding cache {out_path} already exists. Skipping.")
            continue
            
        print(f"Precomputing embeddings for {tag} ({chunk_path})...")
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            
        texts = [f"{c['paper_title']} | Section: {c['section']} | {c['text']}" for c in chunks]
        emb_gen = model.embed(texts)
        embs = np.array(list(emb_gen), dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-12)
        
        np.save(out_path, embs)
        print(f"  Saved {out_path} ({embs.shape})")

if __name__ == "__main__":
    cache_embeddings()

