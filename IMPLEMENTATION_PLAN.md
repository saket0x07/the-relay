# Implementation Plan — The Relay Engine

This step-by-step plan details the exact build sequence to transition from raw PDFs and specifications to a fully verified, evaluated, and measured system complying with the assignment brief.

---

## Phase 1: Environment & Corpus Processing
- [ ] **1.1 Install dependencies**
  - Verify Python 3.12 environment with `fastapi`, `uvicorn`, `httpx`, `rank-bm25`, `fastembed`, `pypdf`, `pandas`, `pyyaml`.
- [ ] **1.2 PDF Document Ingestion (`relay/retrieval/ingest.py`)**
  - Parse the 3 raw PDFs: `1706.03762v7_attention_is_all_you_need.pdf`, `2410.00037v1_moshi.pdf`, and `2507.07955v2_dynamic_chunking.pdf`.
  - Extract structured records preserving: `paper_id`, `section_title`, `page_number`, `chunk_type` (body / table / caption / reference).
  - Isolate reference sections so they can be filtered/downweighted.
  - Save structured canonical JSON in `corpus/processed/corpus_canonical.json`.
- [ ] **1.3 Chunking Strategies & Vector Indexing (`relay/retrieval/chunkers.py`, `dense.py`, `bm25.py`)**
  - Implement 4 settings:
    - **R1:** ~200 tokens, fetch 8, no rerank.
    - **R2:** ~400 tokens, 15% overlap, fetch 5, no rerank (Baseline).
    - **R3:** ~400 tokens, 20 candidates, top 5 reranked.
    - **R4:** ~800 tokens, fetch 3, no rerank.
  - Generate BM25 tokenized indices and FastEmbed (`bge-small-en-v1.5`) dense matrices.

---

## Phase 2: Groq Probing & Stand-In Mock Server
- [ ] **2.1 Live Groq Probing Script (`standin/probe_groq.py`)**
  - Run small, strictly budgeted test calls to Groq API using user key.
  - Capture real response headers: `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, `x-ratelimit-limit-tokens`, `x-ratelimit-remaining-tokens`, `retry-after`.
  - Record TTFT and inter-token distribution under normal conditions.
  - Output findings to `config/limits.yaml` to fill **Table 1** of the brief.
- [ ] **2.2 Calibrated Stand-In Service (`standin/server.py`)**
  - Create standalone FastAPI app on port 8001 simulating `/v1/chat/completions` with SSE streaming.
  - Implement token-bucket throttles for RPM, TPM, and daily caps.
  - Simulate realistic TTFT (log-normal distribution) and inter-token streaming intervals.
  - Implement failure modes: HTTP 429 with `retry-after`, connection resets, and mid-stream pauses.
  - Implement triggerable events: dynamically cut RPM by 50% or zero-out daily budget.

---

## Phase 3: Core Control Plane & Scheduling
- [ ] **3.1 Quota Manager & Token Buckets (`relay/scheduler/quota.py`)**
  - Thread-safe / asyncio token bucket implementation for RPM, TPM, RPD.
  - Two-phase reservation: reserve estimated tokens, reconcile with actual usage from streaming chunk metadata.
  - Parse headers from live responses and adjust bucket state dynamically.
- [ ] **3.2 Adaptive Concurrency & Priority Queue (`relay/scheduler/aimd.py`, `queue.py`)**
  - AIMD concurrency controller (+1 on consecutive successes, half on 429/timeout).
  - Least-Slack-First queue ordering with separate class lanes (short factual, multi-hop, loop-suspect).
  - Request deadline enforcement and pre-spend shedding.
- [ ] **3.3 Degradation Ladder & Resilience (`relay/scheduler/ladder.py`, `breaker.py`, `retry.py`)**
  - Ladder states L0 → L5 triggered by utilization, bucket exhaustion, or queue latency.
  - Circuit breaker transitioning Closed → Open → Half-Open.
  - Exponential backoff with jitter and retry budget limiter (max 15% of total requests).

---

## Phase 4: Data Plane Pipeline (S0–S7) & Tools
- [ ] **4.1 Retrieval Engine & Tools (`relay/retrieval/fusion.py`, `relay/pipeline/tools/`)**
  - Reciprocal Rank Fusion (RRF, $k=60$) combining BM25 and dense vector results.
  - Tool 1: `get_section(paper, section_or_page)` reading from canonical store.
  - Tool 2: `lookup_table(paper, table_id)` returning clean tabular cell data.
- [ ] **4.2 Pipeline State Machine (`relay/pipeline/engine.py` & `steps/`)**
  - **S0:** Cache check on verified queries (zero tokens spent on hit).
  - **S1:** Query understand / rewrite / paper hint extraction.
  - **S2:** Hybrid retrieval (RRF).
  - **S3:** Grade evidence sufficiency (triggers **Loop A** query expansion if insufficient, max 2 iterations).
  - **S4:** Pack context with deduplication and token trimming based on degradation level.
  - **S5:** Draft & Cite streaming tokens via SSE with citation enforce.
  - **S6:** Deterministic + LLM verify (triggers **Loop B** regeneration if failed, max 1 iteration).
  - **S7:** Finalize, trace emission, cache write.
- [ ] **4.3 Telemetry & Logging (`relay/telemetry/`)**
  - Granular span timing written to `data/traces.jsonl`.
  - Canonical row writer appending to `data/results.csv` following brief's exact column format.

---

## Phase 5: API Gateway & SSE Interface
- [ ] **5.1 FastAPI Gateway (`relay/api/gateway.py`)**
  - Streaming SSE endpoint `/api/ask` returning tokens in real time.
  - Admission assignment of request class, deadlines, and tracking IDs.

---

## Phase 6: Load Testing, Evaluation & Table Generation
- [ ] **6.1 Question Evaluation Suite (`eval/questions.yaml`, `eval/runner.py`)**
  - Formulate $\ge 25$ curated test questions:
    - Single-paper questions (e.g. BLEU scores, Moshi latency, Dynamic Chunking boundaries).
    - Multi-hop & cross-paper disambiguation questions (e.g. "chunking" in Moshi vs. Dynamic Chunking).
    - Ambiguous & unanswerable trick questions (testing strict citation and "papers do not say" refusal).
- [ ] **6.2 Retrieval Sweep (`eval/sweep.py`)**
  - Run the test suite across settings R1, R2, R3, R4.
  - Output metrics into **Table 3** (accuracy %, citation fidelity, TTFT, tokens per answer).
- [ ] **6.3 Load Generator Scenarios (`loadgen/runner.py`)**
  - Run the 5 required benchmark scenarios:
    1. `steady`
    2. `burst` (10x step)
    3. `sustained overload`
    4. `per-minute limit halved`
    5. `daily budget exhausted`
  - Output results directly into `results.csv`.
- [ ] **6.4 Analysis & Output Table Builder (`analysis/make_tables.py`)**
  - Verify every single figure in Table 1, Table 2a, Table 2b, and Table 3 matches values computed from `results.csv`.
- [ ] **6.5 Trace Latency Deconstruction**
  - Pick one slow request and decompose latency into `waiting`, `working`, and `retrying` (before vs. after first token) for Section 4 of the report.

---

## Phase 7: Final Documentation & 2-Page Brief Write-Up
- [ ] **7.1 Compile 2-Page Final Submission (`REPORT.md`)**
  - All mandatory headings in exact order (1 through 8 + Retrieval + Pitfalls + Vaani Feedback + Assistant Collaboration).
  - Embed completed Tables 1, 2a, 2b, 3.

