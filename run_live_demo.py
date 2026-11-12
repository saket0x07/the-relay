import asyncio
import json
import time
from relay.backends.groq import GroqBackend
from relay.pipeline.engine import PipelineContext, PipelineEngine
from relay.retrieval.fusion import HybridRetriever
from relay.scheduler.quota import QuotaManager
from relay.telemetry.csv_writer import ResultsLogger

async def run_direct_demo():
    print("Initializing The Relay engine with Groq backend...")
    with open("corpus/processed/chunks_r2_r3.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    retriever = HybridRetriever(chunks)
    backend = GroqBackend()
    quota = QuotaManager(rpm_limit=1000, tpm_limit=8000, rpd_limit=14400, max_inflight=5)
    logger = ResultsLogger()
    engine = PipelineEngine(retriever, backend, quota, logger)

    test_queries = [
        "What is the dimensionality d_model in Attention Is All You Need?",
        "What BLEU score did the big Transformer model get on English-to-German in Table 2?",
        "What is the cafeteria lunch price at the authors university?"  # Unanswerable trap
    ]

    for idx, q in enumerate(test_queries):
        print(f"\n==================================================")
        print(f"QUERY {idx+1}: {q}")
        print(f"==================================================")
        
        ctx = PipelineContext(
            request_id=f"groq_{idx+1:03d}",
            query=q,
            scenario="live_groq",
            source="real",
            setting="R2"
        )

        print("STREAMING ANSWER: ", end="", flush=True)
        async for token in engine.execute(ctx):
            print(token, end="", flush=True)
            
        print(f"\n\n[STATUS: {ctx.status} | GRADE: {ctx.answer_grade} | LEVEL: {ctx.degradation_level}]")
        print(f"Total time: {((time.time() - ctx.start_time)*1000):.1f} ms | Tokens In: {ctx.total_tokens_in} | Tokens Out: {ctx.total_tokens_out}")

if __name__ == "__main__":
    asyncio.run(run_direct_demo())

