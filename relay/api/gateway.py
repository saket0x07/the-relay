"""
FastAPI SSE Gateway for The Relay.
Exposes:
- POST /api/ask : Streaming SSE response with citation tracking
- GET /api/metrics : Live quota utilization and degradation level
- GET /api/health : Health check
"""

import asyncio
import json
import time
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from relay.backends.groq import GroqBackend
from relay.pipeline.engine import PipelineContext, PipelineEngine
from relay.retrieval.fusion import HybridRetriever
from relay.scheduler.quota import QuotaManager
from relay.telemetry.csv_writer import ResultsLogger

app = FastAPI(title="The Relay Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine Initialization
print("Loading canonical chunk indices for The Relay...")
with open("corpus/processed/chunks_r2_r3.json", "r", encoding="utf-8") as f:
    default_chunks = json.load(f)

retriever = HybridRetriever(default_chunks)
backend = GroqBackend()
quota = QuotaManager(rpm_limit=1000, tpm_limit=8000, rpd_limit=14400, max_inflight=5)
logger = ResultsLogger()
engine = PipelineEngine(retriever, backend, quota, logger)
req_counter = 0


class AskRequest(BaseModel):
    query: str
    retrieval_setting: Optional[str] = "R2"
    scenario: Optional[str] = "interactive"


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "The Relay", "timestamp": time.time()}


@app.get("/api/metrics")
async def metrics():
    snapshot = quota.get_snapshot()
    return {
        "quota": snapshot,
        "verified_cache_entries": len(engine.verified_cache)
    }


@app.post("/api/ask")
async def ask(req: AskRequest):
    global req_counter
    req_counter += 1
    req_id = f"ask_{int(time.time()*1000)}_{req_counter:04d}"

    ctx = PipelineContext(
        request_id=req_id,
        query=req.query,
        scenario=req.scenario or "interactive",
        source="openrouter",
        setting=req.retrieval_setting or "R2"
    )

    async def event_generator():
        # First chunk transmits request ID and status
        init_meta = {
            "type": "metadata",
            "request_id": req_id,
            "status": "admitted",
            "setting": ctx.retrieval_setting
        }
        yield f"data: {json.dumps(init_meta)}\n\n"

        async for token in engine.execute(ctx):
            token_event = {"type": "token", "delta": token}
            yield f"data: {json.dumps(token_event)}\n\n"

        final_meta = {
            "type": "done",
            "request_id": req_id,
            "status": ctx.status,
            "degradation_level": ctx.degradation_level,
            "spans": ctx.step_spans,
            "tokens_in": ctx.total_tokens_in,
            "tokens_out": ctx.total_tokens_out
        }
        yield f"data: {json.dumps(final_meta)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("relay.api.gateway:app", host="127.0.0.1", port=8000, reload=False)

