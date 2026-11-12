# System Architecture & Technical Specifications — The Relay

> **Project:** The Relay — Scalable, Quota-Aware RAG Pipeline with Hybrid Search  
> **Corpus:** 3 arXiv Papers (`1706.03762v7`, `2410.00037v1`, `2507.07955v2`)  
> **Execution Platform:** Python 3.12, `asyncio`, `FastAPI`, `httpx`  
> **LLM Layer:** Groq Free Tier (`llama-3.1-8b-instant` / `llama-3.3-70b-versatile`) + Calibrated Stand-In Mock Server

---

## 1. System Overview & The Core Tension

```
   Bursty Incoming Requests (10x spikes, sustained overloads)
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   THE RELAY: ADMISSION & CONTROL PLANE                 │
│  - Non-blocking SSE gateway with monotonic request IDs                 │
│  - Adaptive Concurrency (AIMD) & Slack-based priority scheduling       │
│  - Global Quota Manager (RPM, TPM, RPD, in-flight tracking)            │
│  - 6-Stage Graceful Degradation Ladder (L0 -> L5)                      │
│  - Circuit Breaker & Retry Budget (~15% cap)                           │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         8-STEP PIPELINE ENGINE                         │
│  S0: Admit & Verified Cache Check ──[Hit]──────────────────────────┐   │
│  S1: Query Understanding & Rewrite                                 │   │
│  S2: Hybrid Retrieval (Local BM25 + FastEmbed Dense + RRF) ◄───┐   │   │
│  S3: Grade Evidence ──[Insufficient]───────────────────────────┘   │   │
│      ├── Tool 1: get_section(paper, sec)                           │   │
│      └── Tool 2: lookup_table(paper, table_id)                     │   │
│  S4: Pack & Deduplicate Context                                    │   │
│  S5: Draft & Cite (Streamed LLM tokens) ◄──────────────────────┐   │   │
│  S6: Verify (Deterministic Check + Entailment) ──[Fail]────────┘   │   │
│  S7: Finalize, Emit Telemetry, Cache Answer ◄──────────────────────┘   │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
         Unified Results Logging (`results.csv`) & Trace Telemetry
```

### The Core Tension
In high-throughput RAG systems on free or fixed API tiers:
1. **The LLM quota is non-scalable**: Adding worker processes does not add API quota.
2. **Context size drives throughput**: Since TPM (Tokens Per Minute) binds earlier than RPM (Requests Per Minute), context window length is a primary capacity throttle.
3. **Correctness vs. Speed**: Our invariant is: **Never ship an unverified, fabricated, or uncited claim to save time.** When quota fails or degrades, the system falls back to extractive citations or honest refusals rather than guessing.

---

## 2. Directory Layout & Separation of Concerns

```
d:\Projects\VAANI\
├── config/                     # Configuration definitions
│   ├── limits.yaml             # Real Groq quotas + fallback limits
│   ├── ladder.yaml             # Thresholds & degradation rules (L0-L5)
│   ├── retrieval.yaml          # Chunk sizes, weights, RRF parameters (R1-R4)
│   ├── scenarios.yaml          # Arrival rates, shapes (steady, burst, overload)
│   └── standin.yaml            # Latency distributions, error injection rates
├── corpus/
│   ├── raw/                    # Downloaded PDFs (SHA-256 verified)
│   ├── processed/              # Extracted sections, tables, canonical sentences
│   ├── indices/                # Pre-built BM25 & dense vector embeddings
│   └── manifest.json           # Integrity metadata
├── relay/                      # PRODUCTION RUNTIME (Zero dependency on test code)
│   ├── api/
│   │   ├── gateway.py          # FastAPI routes, SSE streaming
│   │   └── admission.py        # Token estimation, class tagging, deadline assignment
│   ├── scheduler/
│   │   ├── quota.py            # Token buckets (RPM, TPM, RPD, TPD), header reconciliation
│   │   ├── aimd.py             # Additive Increase Multiplicative Decrease concurrency
│   │   ├── queue.py            # Priority Queue (Least Slack First) with class isolation
│   │   ├── ladder.py           # Degradation Controller (L0-L5 transitions)
│   │   ├── breaker.py          # Circuit Breaker (Open, Half-Open, Closed)
│   │   └── retry.py            # Exponential backoff with jitter & retry budget
│   ├── pipeline/
│   │   ├── engine.py           # Execution state machine
│   │   ├── context.py          # Pipeline request state container
│   │   ├── steps/              # Individual steps S0 - S7
│   │   │   ├── s0_admit_cache.py
│   │   │   ├── s1_understand.py
│   │   │   ├── s2_retrieve.py
│   │   │   ├── s3_grade.py
│   │   │   ├── s4_pack.py
│   │   │   ├── s5_draft.py
│   │   │   ├── s6_verify.py
│   │   │   └── s7_finalize.py
│   │   └── tools/
│   │       ├── get_section.py  # Canonical text reader
│   │       └── lookup_table.py # Structural table parser
│   ├── retrieval/
│   │   ├── ingest.py           # PyPDF parser with section & table bounding
│   │   ├── chunkers.py         # 200 / 400 / 800 token chunking strategies
│   │   ├── bm25.py             # BM25 engine with technical symbol preservation
│   │   ├── dense.py            # FastEmbed local ONNX embedding generator
│   │   └── fusion.py           # Reciprocal Rank Fusion (RRF, k=60)
│   ├── backends/
│   │   ├── base.py             # LLMBackend Protocol definition
│   │   ├── groq.py             # Real Groq HTTPX streaming client
│   │   └── standin_client.py   # Client connecting to standin mock
│   └── telemetry/
│       ├── trace.py            # Per-step granular JSONL spans
│       └── csv_writer.py       # results.csv thread-safe append writer
├── standin/                    # STAND-IN MOCK SERVICE (Port 8001)
│   ├── server.py               # OpenAI/Groq compatible /v1/chat/completions SSE server
│   ├── distributions.py        # Latency sampling (fitted Weibull / Log-Normal)
│   └── fault_injector.py       # Dynamic limits, 429s, mid-run cutoffs
├── loadgen/                    # EVALUATION & LOAD GENERATOR
│   ├── arrivals.py             # Poisson, burst step, sustained ramps
│   ├── scenarios.py            # Scenario definitions (steady, burst, halved, daily dead)
│   └── runner.py               # CLI runner orchestrating benchmark runs
├── eval/                       # ACCURACY & CORPUS EVALUATION
│   ├── questions.yaml          # >=15 curated questions (single-paper, numeric, traps)
│   ├── runner.py               # Test runner against real/standin backends
│   └── grade.py                # Automated & rule-based citation verifier
└── analysis/                   # BRIEF REPORT GENERATION
    └── make_tables.py          # Generates Table 1, 2a, 2b, 3 directly from results.csv
```

---

## 3. Detailed Component Architecture

### 3.1 Control Plane: Quota, Concurrency, and Degradation
1. **Quota Manager (`relay/scheduler/quota.py`):**
   - Implements 4 concurrent token buckets:
     - `RPM`: 1-minute rolling request rate.
     - `TPM`: 1-minute rolling token rate.
     - `RPD`: 24-hour request limit.
     - `TPD`: 24-hour token budget.
   - **Two-phase reservation:** When starting an LLM call, it reserves `est_input_tokens + max_completion_tokens`. On stream completion, it reconciles the bucket with `actual_tokens` (issuing refunds or recording deficits).
   - **Header-driven learning:** Inspects response headers `x-ratelimit-remaining-tokens`, `x-ratelimit-remaining-requests`, `x-ratelimit-reset-tokens`, `retry-after` to adjust state dynamically.

2. **Adaptive Concurrency (`relay/scheduler/aimd.py`):**
   - Controls in-flight LLM calls.
   - Starts with `concurrency = 2`.
   - On success burst (e.g. 10 consecutive passes): `concurrency += 1` (capped at max configured).
   - On 429 or slow timeout: `concurrency = max(1, floor(concurrency * 0.5))`.

3. **Priority Queue (`relay/scheduler/queue.py`):**
   - Employs **Least Slack First**: `slack = deadline - now - estimated_remaining_time`.
   - Dedicated lanes for short questions, multi-hop queries, and loop-suspect requests to avoid queue starvation.
   - Early shedding: If a request in queue has `slack < 0` before making any LLM call, it drops directly to extractive fallback or polite timeout notice without consuming provider quota.

4. **Degradation Controller (`relay/scheduler/ladder.py`):**
   | Level | Condition | Actions & Fallbacks | Guarantee |
   |---|---|---|---|
   | **L0** | Quota Util < 60% | Full pipeline: Large model (`llama-3.3-70b`), Rerank on, k=5 chunks | Peak Quality |
   | **L1** | Util 60%–80% | Skip reranker, k=4 chunks, stream immediately | Fast TTFT |
   | **L2** | Util 80%–95% | Small model (`llama-3.1-8b`), rule-based S1 query rewrite | Low Token Cost |
   | **L3** | Util > 95% or Queue Lagging | Single retrieval pass (no Loop A), strictly budgeted generation | Zero Queue Stall |
   | **L4** | Breaker Open or RPD Exhausted | Extractive-only response (top retrieved passages + citations, no LLM) | Zero Quota Spent |
   | **L5** | Extractive confidence low | Honest refusal ("Cannot answer at this time; quota resets at HH:MM") | Never Fabricate |

---

### 3.2 Data Plane: 8-Step Pipeline & Tools
The engine runs a state machine passing a `PipelineContext` object through 8 sequential steps with branching and loop capabilities:

- **S0: Admit & Cache Check (`s0_admit_cache.py`)**
  - Normalizes query string.
  - Queries local in-memory exact/semantic verified cache.
  - *Branch B1*: Cache hit returns validated answer with `0` tokens consumed directly to S7.
- **S1: Query Understanding (`s1_understand.py`)**
  - Resolves paper targeting (Attention vs. Moshi vs. Dynamic Chunking).
  - Resolves ambiguous technical terms (e.g. "chunking" in RAG vs "chunking" in dynamic hierarchical modeling).
  - Decomposes complex queries into sub-queries. Fallback to heuristic parser under L2+.
- **S2: Hybrid Retrieval (`s2_retrieve.py`)**
  - Dual lookup: BM25 (token-preserved keywords) + FastEmbed ONNX vectors.
  - Merged via Reciprocal Rank Fusion ($RRF\_score = \sum \frac{1}{60 + rank}$).
  - Soft paper-boost applied if S1 specified a paper hint.
- **S3: Grade Evidence (`s3_grade.py`)**
  - Computes top candidate relevance score.
  - *Branch B2*: If score < threshold, trigger **Loop A** (expand query, maximum 2 iterations). If loops exhausted and still absent, jump to refusal template.
  - *Branch B3*: If question is numeric / tabular, call **Tool 2: `lookup_table`**. If page/section boundaries are cut off, call **Tool 1: `get_section`**.
- **S4: Pack Context (`s4_pack.py`)**
  - Deduplicates overlapping spans, stitches adjacent text blocks.
  - Trims passages to fit current degradation level token budget.
  - *Branch B4*: If at L4/L5, package top passages as extractive answer and branch directly to S7.
- **S5: Draft & Cite (`s5_draft.py`)**
  - Calls `LLMBackend.stream()`.
  - Streams tokens via SSE to the user.
  - Mandates citation format: `[Paper ID, Section X.Y, Page Z]`.
- **S6: Verify (`s6_verify.py`)**
  - Step 1 (Deterministic): Verifies numbers, table values, and named entities match retrieved text.
  - Step 2 (Entailment): Small model check or heuristic rule verifying claims are substantiated.
  - *Branch B5*: If verification fails, trigger **Loop B** (regenerate with strict constraint prompt, max 1 iteration). If it fails twice, downgrade to extractive answer or refusal.
- **S7: Finalize (`s7_finalize.py`)**
  - Commits verified answer to cache.
  - Writes granular step durations to `traces.jsonl`.
  - Appends canonical row to `results.csv`.

---

### 3.3 Hybrid Retrieval Engine
- **Corpus Tokenization:** Standard tokenizers destroy hyphenated terms like `d_model`, `dim_feedforward`, `rq-vae`, `v1.0`. Custom regex preserves technical jargon.
- **Dense Vector Search:** FastEmbed (`BAAI/bge-small-en-v1.5`) running locally in ONNX runtime. Zero API quota used, executes in <10ms.
- **Fusion:** RRF with constant $k=60$. Combines BM25 and dense rank lists without requiring score calibration.
- **Sweep Settings (R1–R4):**
  - **R1:** ~200 tokens/chunk, fetch 8 pieces, no rerank.
  - **R2:** ~400 tokens/chunk (paragraph-aware, 15% overlap), fetch 5 pieces, no rerank.
  - **R3:** ~400 tokens/chunk, fetch 20 candidates, rerank top 5.
  - **R4:** ~800 tokens/chunk, fetch 3 pieces, no rerank.

---

### 3.4 Stand-In Mock Service & Calibration
- Implements `/v1/chat/completions` supporting streaming SSE (`text/event-stream`).
- Calibrated to match real Groq free tier distributions:
  - **TTFT (Time To First Token):** Log-normal distribution ($μ \approx 250\text{ms}, σ \approx 0.35$).
  - **Inter-token Latency:** Uniform/Normal distribution ($10\text{ms} - 25\text{ms}$ per token).
  - **Load multiplier:** $latency = base \times (1 + \alpha \cdot (\frac{inflight}{capacity})^\beta)$.
  - **Simulated 429:** Returns HTTP 429 with headers `retry-after: 2.5`, `x-ratelimit-remaining-tokens: 0`.
  - **Scripted Events:** Mid-run capacity cuts (50% RPM reduction) and daily exhaustion triggers.

---

### 3.5 Evaluation & Results Generation
- **`results.csv` Contract:**
  ```csv
  request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
  ```
- **Automated Tables (`analysis/make_tables.py`):**
  - **Table 1:** Real service discovered limits (RPM, TPM, RPD, in-flight, recovery).
  - **Table 2a:** Latency percentiles (p50, p95, p99) for 5 scenarios (steady, burst, sustained overload, limit halved, daily budget exhausted).
  - **Table 2b:** Outcomes (succeeded %, 429 %, timeout %, retries, queue length, tokens/answer, cost/1k).
  - **Table 3:** Retrieval sweep comparing R1, R2, R3, R4 across accuracy, citation fidelity, and latency.

---

## 4. End-to-End Execution Sequence

```
1. Ingest Corpus (3 PDFs -> clean sections, tables, JSON manifest)
2. Build Embeddings & BM25 indices (R1-R4 pre-chunked variants)
3. Probe Groq API (discover true RPM, TPM, headers, latency profile)
4. Configure & Start Stand-In Mock Server (matching probe parameters)
5. Build Relay Core Engine (Pipeline, Queue, AIMD, Degradation Ladder)
6. Execute Retrieval Benchmark Sweep (populate Table 3)
7. Execute Load Scenarios (Steady, Burst, Overload, Halved, Exhausted)
8. Run Analysis Script (Generate Table 1, 2a, 2b & results.csv)
9. Compile Final 2-Page Brief Report
```

