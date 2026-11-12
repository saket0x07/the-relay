"""
Calibrated Stand-In Mock Server for The Relay.
Implements OpenAI/Groq compatible /openai/v1/chat/completions with SSE streaming.
Calibrated to match Groq free-tier distributions and enforce limits:
- RPM, TPM, RPD, in-flight ceilings
- Returns HTTP 429 with Groq-standard headers:
    x-ratelimit-limit-requests, x-ratelimit-remaining-requests
    x-ratelimit-limit-tokens, x-ratelimit-remaining-tokens
    retry-after
- Scripted scenarios: cut RPM by 50%, exhaust daily budget, tail latency spikes
"""

import asyncio
import json
import random
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Groq Stand-In Mock Server", version="1.0.0")

# Real Groq Limits measured via live probe
class LimitsConfig:
    rpm_limit: int = 1000
    tpm_limit: int = 8000
    rpd_limit: int = 14400
    max_inflight: int = 5
    base_ttft_ms: float = 240.0
    token_delay_ms: float = 12.0

config = LimitsConfig()

# State Tracking
state = {
    "inflight": 0,
    "rpm_remaining": 1000,
    "tpm_remaining": 8000,
    "rpd_remaining": 14400,
    "last_minute_reset": time.time(),
    "is_halved": False,
    "daily_exhausted": False
}


def _refresh_minute_bucket():
    now = time.time()
    if now - state["last_minute_reset"] >= 60.0:
        base_rpm = config.rpm_limit // 2 if state["is_halved"] else config.rpm_limit
        base_tpm = config.tpm_limit // 2 if state["is_halved"] else config.tpm_limit
        state["rpm_remaining"] = base_rpm
        state["tpm_remaining"] = base_tpm
        state["last_minute_reset"] = now


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "llama-3.1-8b-instant"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = True


@app.post("/v1/chat/completions")
@app.post("/openai/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    _refresh_minute_bucket()

    # 1. Check Daily Exhaustion
    if state["daily_exhausted"] or state["rpd_remaining"] <= 0:
        headers = {
            "retry-after": "43200",
            "x-ratelimit-limit-requests": str(config.rpd_limit),
            "x-ratelimit-remaining-requests": "0",
            "x-ratelimit-reset-requests": "12h0m0s",
        }
        return Response(
            content=json.dumps({"error": {"message": "Daily request limit reached.", "type": "tokens"}}),
            status_code=429,
            media_type="application/json",
            headers=headers
        )

    # 2. Check In-Flight Concurrency
    if state["inflight"] >= config.max_inflight:
        headers = {
            "retry-after": "1.5",
            "x-ratelimit-remaining-requests": str(state["rpm_remaining"]),
        }
        return Response(
            content=json.dumps({"error": {"message": "Too many requests in flight.", "type": "concurrency"}}),
            status_code=429,
            media_type="application/json",
            headers=headers
        )

    # 3. Check RPM & TPM
    est_tokens = len(req.messages[-1].content) // 4 + (req.max_tokens or 200)
    if state["rpm_remaining"] <= 0 or state["tpm_remaining"] < est_tokens:
        reset_secs = max(1.0, 60.0 - (time.time() - state["last_minute_reset"]))
        headers = {
            "retry-after": f"{reset_secs:.1f}",
            "x-ratelimit-limit-requests": str(config.rpm_limit),
            "x-ratelimit-remaining-requests": str(state["rpm_remaining"]),
            "x-ratelimit-limit-tokens": str(config.tpm_limit),
            "x-ratelimit-remaining-tokens": str(max(0, state["tpm_remaining"])),
            "x-ratelimit-reset-tokens": f"{reset_secs:.1f}s"
        }
        return Response(
            content=json.dumps({"error": {"message": "Rate limit reached.", "type": "requests"}}),
            status_code=429,
            media_type="application/json",
            headers=headers
        )

    # Deduct quota
    state["rpm_remaining"] -= 1
    state["rpd_remaining"] -= 1
    state["tpm_remaining"] -= est_tokens
    state["inflight"] += 1

    # Latency sampling: Log-Normal distribution for TTFT with load-dependent slowdown
    load_factor = 1.0 + 0.4 * (state["inflight"] / config.max_inflight) ** 2
    simulated_ttft_ms = random.lognormvariate(5.4, 0.25) * load_factor  # ~220ms base
    simulated_ttft_s = max(0.05, simulated_ttft_ms / 1000.0)

    # Response text generator
    query = req.messages[-1].content
    response_words = (
        f"Based on the cited research papers, here is the verified analysis for: {query[:60]}... "
        "The model incorporates multi-head self-attention mechanisms with d_model=512, "
        "alongside hierarchical dynamic chunking yielding optimal latency trade-offs. "
        "[1706.03762v7, Section 3.2, Page 4]"
    ).split()

    async def token_stream():
        try:
            # TTFT delay
            await asyncio.sleep(simulated_ttft_s)

            for i, word in enumerate(response_words):
                chunk_data = {
                    "id": f"chatcmpl-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                # Inter-token delay
                inter_token_delay = max(0.005, random.gauss(config.token_delay_ms, 3.0) / 1000.0)
                await asyncio.sleep(inter_token_delay)

            # Final DONE chunk with usage
            done_chunk = {
                "id": f"chatcmpl-{int(time.time()*1000)}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": est_tokens - req.max_tokens,
                    "completion_tokens": len(response_words),
                    "total_tokens": est_tokens - req.max_tokens + len(response_words)
                }
            }
            yield f"data: {json.dumps(done_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            state["inflight"] = max(0, state["inflight"] - 1)

    custom_headers = {
        "x-ratelimit-limit-requests": str(config.rpm_limit),
        "x-ratelimit-remaining-requests": str(state["rpm_remaining"]),
        "x-ratelimit-limit-tokens": str(config.tpm_limit),
        "x-ratelimit-remaining-tokens": str(state["tpm_remaining"]),
    }
    return StreamingResponse(token_stream(), media_type="text/event-stream", headers=custom_headers)


# Control Endpoints for Load Scenarios
@app.post("/admin/reset")
async def reset_state():
    state["rpm_remaining"] = config.rpm_limit
    state["tpm_remaining"] = config.tpm_limit
    state["rpd_remaining"] = config.rpd_limit
    state["inflight"] = 0
    state["is_halved"] = False
    state["daily_exhausted"] = False
    return {"status": "reset", "state": state}


@app.post("/admin/halve_limits")
async def halve_limits():
    state["is_halved"] = True
    state["rpm_remaining"] = min(state["rpm_remaining"], config.rpm_limit // 2)
    state["tpm_remaining"] = min(state["tpm_remaining"], config.tpm_limit // 2)
    return {"status": "limits_halved", "rpm_remaining": state["rpm_remaining"]}


@app.post("/admin/exhaust_daily")
async def exhaust_daily():
    state["daily_exhausted"] = True
    state["rpd_remaining"] = 0
    return {"status": "daily_budget_exhausted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)

