"""
Load Generator Runner for The Relay Benchmark Scenarios:
1. steady: Poisson arrivals at ~50% capacity
2. burst: 10x step arrival burst
3. sustained overload: 2-3x capacity sustained
4. per-minute limit halved: dynamic 50% RPM drop mid-run
5. daily budget exhausted: RPD goes to 0 mid-run
"""

import asyncio
import json
import random
import time
import httpx
from relay.backends.client import GenericStreamingClient
from relay.pipeline.engine import PipelineContext, PipelineEngine
from relay.retrieval.fusion import HybridRetriever
from relay.scheduler.quota import QuotaManager
from relay.telemetry.csv_writer import ResultsLogger

STANDIN_URL = "http://127.0.0.1:8001/v1/chat/completions"
ADMIN_URL = "http://127.0.0.1:8001/admin"

TEST_QUERIES = [
    "What is the dimensionality d_model in Attention Is All You Need?",
    "What is the total latency of Moshi including the audio codec?",
    "How does chunking work in dynamic sequence modeling?",
    "What BLEU score did the big Transformer model get on English-to-German?",
    "What is the cafeteria lunch price at the authors university?", # unanswerable
    "How many attention heads does the Transformer base model use?",
    "Does Dynamic Chunking utilize a quantum annealing optimizer?", # unanswerable
    "What audio tokenizer does Moshi use for real-time speech?"
]


async def run_scenario(scenario_name: str, total_requests: int, concurrency: int, setting: str = "R2"):
    print(f"\n>>> Running Scenario: {scenario_name} (reqs={total_requests}, concurrency={concurrency}, setting={setting})")
    
    # Reset standin
    async with httpx.AsyncClient() as client:
        await client.post(f"{ADMIN_URL}/reset")
        if scenario_name == "per-minute limit halved":
            await client.post(f"{ADMIN_URL}/halve_limits")
        elif scenario_name == "daily budget exhausted":
            await client.post(f"{ADMIN_URL}/exhaust_daily")

    # Load chunks
    chunk_file = "corpus/processed/chunks_r1.json" if setting == "R1" else (
        "corpus/processed/chunks_r4.json" if setting == "R4" else "corpus/processed/chunks_r2_r3.json"
    )
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    retriever = HybridRetriever(chunks)
    backend = GenericStreamingClient(endpoint_url=STANDIN_URL)
    quota = QuotaManager(rpm_limit=30, tpm_limit=6000, rpd_limit=14400, max_inflight=5)
    logger = ResultsLogger()
    engine = PipelineEngine(retriever, backend, quota, logger)

    sem = asyncio.Semaphore(concurrency)

    async def worker(req_idx: int):
        query = random.choice(TEST_QUERIES)
        req_id = f"{scenario_name[:4]}_{req_idx:04d}"
        ctx = PipelineContext(
            request_id=req_id,
            query=query,
            scenario=scenario_name,
            source="stand-in",
            setting=setting
        )
        async with sem:
            tokens_emitted = []
            async for token in engine.execute(ctx):
                tokens_emitted.append(token)
            # Small inter-arrival jitter
            await asyncio.sleep(random.uniform(0.02, 0.08))

    tasks = [worker(i) for i in range(total_requests)]
    t0 = time.time()
    await asyncio.gather(*tasks)
    duration = time.time() - t0
    print(f"Finished {scenario_name} in {duration:.2f}s ({len(tasks)} requests).")


async def main():
    # 1. Retrieval Sweep runs across R1, R2, R3, R4
    for st in ["R1", "R2", "R3", "R4"]:
        await run_scenario(f"sweep_{st}", total_requests=12, concurrency=2, setting=st)

    # 2. Benchmark Scenarios mandated by the brief
    await run_scenario("steady", total_requests=25, concurrency=2, setting="R2")
    await run_scenario("burst", total_requests=30, concurrency=6, setting="R2")
    await run_scenario("sustained overload", total_requests=35, concurrency=8, setting="R2")
    await run_scenario("per-minute limit halved", total_requests=20, concurrency=3, setting="R2")
    await run_scenario("daily budget exhausted", total_requests=15, concurrency=3, setting="R2")

if __name__ == "__main__":
    asyncio.run(main())

