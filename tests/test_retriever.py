import json
from relay.retrieval.fusion import HybridRetriever

def smoke_test():
    with open('corpus/processed/chunks_r2_r3.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")
    retriever = HybridRetriever(chunks)
    
    query = "What is the value of d_model in the base Transformer?"
    print(f"\nQuery: {query}")
    results = retriever.search(query, k_final=3)
    
    for idx, res in enumerate(results):
        print(f"Hit {idx+1}: [{res['paper_id']} | {res['section']} | p. {res['page_str']}] (Score: {res['rrf_score']:.4f})")
        print("  Snippet:", res['text'][:140].replace("\n", " "), "...\n")

if __name__ == "__main__":
    smoke_test()

