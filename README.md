# The Relay: A Scalable, Quota-Aware RAG Pipeline with Hybrid Search

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq Cloud](https://img.shields.io/badge/Groq-Cloud%20API-orange.svg)](https://groq.com)

**The Relay** is a production-grade, quota-aware Retrieval-Augmented Generation (RAG) system engineered to operate predictably under stringent API provider limits (RPM, TPM, RPD, and in-flight concurrency) while enforcing a zero-compromise invariant: **never guess, never hallucinate numbers, and never ship an uncited claim.**

Built specifically for high-throughput engineering question answering over three pinned arXiv research papers:
1. *"Attention Is All You Need"* (`arXiv:1706.03762v7`)
2. *"Moshi: a speech-text foundation model for real-time dialogue"* (`arXiv:2410.00037v1`)
3. *"Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"* (`arXiv:2507.07955v2`)

---

## 🎯 Architectural Highlights

### 1. The Core Insight: Quota is a Global Non-Scalable Bottleneck
Adding worker nodes or horizontal consumers does not add API quota. In high-context RAG pipelines (1,000–3,000 tokens/prompt), **Tokens Per Minute (TPM) binds long before Requests Per Minute (RPM)**. Therefore, scalability in The Relay is governed by:
- **Quota Budgeting & Two-Phase Reservation**: Reserving estimated tokens, adjusting dynamically against provider `x-ratelimit-*` headers, and reconciling on stream completion.
- **Context Size as a Throttle**: Scaling context down or shifting degradation tiers rather than failing or dropping requests.
- **Graceful Degradation (L0–L5)**: When overload occurs, requests degrade smoothly to local extractive citations or honest refusals rather than failing with 429 cascades.

```
                           ┌───────────────────────── CONTROL PLANE ─────────────────────────┐
                           │  Quota Manager  ·  AIMD Concurrency  ·  Degradation Controller  │
                           │  Circuit Breaker (Closed/Open/Half-Open) · Retry Budget (≤15%)   │
                           └───────────▲─────────────────────────────────────▲───────────────┘
                                       │ policy / grants                     │ headers, 429s
client ─► API Gateway ─► Admission ────┴─► Priority Deadline Queue ───────► Pipeline Engine
(SSE)     (FastAPI)      (Token budget,    (Least Slack First,               (S0 → S7)
                          deadlines)        class lanes)                           │
                                                                                   ▼
                                                       ┌─────────────────┬─────────────────┬─────────────────┐
                                                       │ Local Hybrid    │ Pluggable       │ Deterministic   │
                                                       │ Retrieval       │ LLM Backend     │ Tool Runner     │
                                                       │ (BM25 + Dense)  │ (Groq/Stand-in) │ (Table/Section) │
                                                       └─────────────────┴─────────────────┴─────────────────┘
                                                                                 │
                                               Telemetry (data/traces.jsonl & data/results.csv) ◄─┘
```

---

## ⚡ 8-Step Pipeline Engine & Tools

The pipeline implements an 8-step state machine with 2 deterministic tools, 5 branches, and 2 recovery loops:

- **S0: Admit & Cache Check**: Normalizes queries and checks a verified answer cache (hit = 0 provider tokens, instant response).
- **S1: Query Understand**: Disambiguates paper targets (e.g. dynamic chunking vs. RAG chunking) and extracts paper soft hints.
- **S2: Local Hybrid Retrieval**:
  - **BM25 Lexical Engine**: Technical-term preserving tokenizer retaining compound symbols (`d_model`, `v1.0`, `1706.03762v7`).
  - **FastEmbed Dense Vectors**: Local `bge-small-en-v1.5` embeddings running in ONNX runtime (**0 API tokens, <15ms latency**).
  - **Reciprocal Rank Fusion (RRF $k=60$)**: Rank-based fusion immune to score calibration discrepancies.
- **S3: Grade Evidence**:
  - Evaluates relevance confidence.
  - **Tool 1 (`lookup_table`)**: Extracts structured tables to prevent PDF OCR damage from causing numerical hallucinations.
  - **Tool 2 (`get_section`)**: Retrieves canonical verbatim text for boundary or page context.
  - **Loop A (Query Expansion)**: Re-retrieves if evidence is insufficient (max 2 iterations). Refuses with template if completely absent.
- **S4: Pack Context**: Deduplicates and stitches text spans, trimming to current degradation token budget.
- **S5: Draft & Cite**: Streams tokens from Groq API via Server-Sent Events (SSE) enforcing citation format `[PaperID, Section, Page]`.
- **S6: Verify**:
  - Deterministic check ensuring citations and numbers appear in source passages.
  - **Loop B (Regeneration)**: Regenerates with strict constraints if citations are missing (max 1 loop).
- **S7: Finalize**: Caches verified answers, writes granular JSONL span traces, and logs rows to `results.csv`.

---

## 🎚️ 6-Stage Graceful Degradation Ladder

| Level  | Condition                            | Pipeline Behavior                                                    | Invariant                  |
| ------ | ------------------------------------ | -------------------------------------------------------------------- | -------------------------- |
| **L0** | Quota Util < 60%                     | Full pipeline: large context, reranker enabled, $k=5$                | Peak synthesis quality     |
| **L1** | Util 60%–80%                         | Skip reranker, fetch $k=4$, immediate SSE stream                     | Fast Time-to-First-Token   |
| **L2** | Util 80%–95%                         | Context compression ($k=4$), rule-based S1 understanding             | Low token footprint        |
| **L3** | Util > 95% or Queue Backlog          | Single retrieval pass (Loop A off), strict completion tokens         | Bounded queue latency      |
| **L4** | Circuit Breaker Open / RPD Exhausted | **Extractive Mode**: top verified passages + citations (0 LLM calls) | Truthful answer shipped    |
| **L5** | Low Retrieval Confidence             | Honest Refusal: *"The papers do not say; quota resets at HH:MM"*     | Never guess or hallucinate |

---

## 📁 Repository Structure

```
the-relay/
├── config/                      # System limits, scenarios, and ladder thresholds
│   ├── limits.yaml              # Measured Groq limits (RPM, TPM, RPD)
│   └── ladder.yaml              # Degradation trigger rules
├── corpus/                      # Paper documents and processed indexes
│   ├── manifest.json            # SHA-256 integrity manifest for arXiv PDFs
│   └── processed/               # Canonical structured JSON & chunk variants (R1-R4)
├── relay/                       # Production Engine Runtime
│   ├── api/                     # FastAPI SSE Streaming Gateway & metrics
│   ├── scheduler/               # QuotaManager, AIMD concurrency, Least-Slack queue
│   ├── pipeline/                # S0–S7 engine and tools (lookup_table, get_section)
│   ├── retrieval/               # BM25, ONNX dense embedder, and RRF fusion
│   ├── backends/                # GroqBackend, OpenRouterBackend, GenericClient
│   └── telemetry/               # Granular span tracer and results CSV writer
├── standin/                     # Calibrated Mock Service for heavy load testing
│   ├── server.py                # SSE /v1/chat/completions server with 429 injection
│   └── probe_groq.py            # Empirical limit probing script
├── loadgen/                     # Synthetic benchmark scenario generator
│   └── runner.py                # Runs steady, burst, sustained overload, halved, exhausted
├── eval/                        # Evaluation Suite
│   ├── questions.yaml           # 16 curated test questions (factual, tables, traps)
│   └── questions.py             # Questions bank builder
├── analysis/                    # Results compilation
│   └── make_tables.py           # Reproduces Tables 1, 2a, 2b, 3 from results.csv
├── tests/                       # Unit and integration tests
├── .env.example                 # Environment configuration template
└── requirements.txt             # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
- Python 3.11 or 3.12
- Git

```bash
git clone https://github.com/saket0x07/the-relay.git
cd the-relay

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment file and insert your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_MODEL=openai/gpt-oss-20b
GROQ_ENDPOINT=https://api.groq.com/openai/v1/chat/completions
```

---

## 💻 Running the System

### A. Run End-to-End Live Demo
Executes interactive queries (factual, table lookup, and trap queries) directly against the Groq Cloud API:
```bash
python run_live_demo.py
```

### B. Launch the Production FastAPI Streaming Gateway
Start the HTTP SSE streaming gateway on port 8000:
```bash
python -m uvicorn relay.api.gateway:app --host 127.0.0.1 --port 8000
```
Query via `curl` with real-time SSE token streaming:
```bash
curl -N -X POST http://127.0.0.1:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the dimensionality d_model in Attention Is All You Need?"}'
```

Check live Quota Manager utilization:
```bash
curl http://127.0.0.1:8000/api/metrics
```

### C. Run Load Benchmark Scenarios & Calibrated Stand-in
To test the pipeline against high-concurrency loads without draining your daily Groq quota:

```bash
# 1. Start the Stand-in mock server (Terminal 1)
python -m uvicorn standin.server:app --port 8001

# 2. Run benchmark scenarios (Terminal 2)
# Runs retrieval sweeps (R1-R4) + steady, burst, sustained overload, limit halved, daily exhausted
python loadgen/runner.py

# 3. Generate summary tables directly from data/results.csv
python analysis/make_tables.py
```

---

## 🧪 Evaluation & Empirical Verification

- **Empirically Discovered Groq Limits**: Probed with sub-second header tracking (`RPM: 1,000`, `TPM: 8,000`, `reset: ~622ms`).
- **Zero Hallucination Traps**: Queries concerning unmentioned facts (e.g., campus cafeteria prices, AWS dollar costs) are deterministically routed to:  
  `"The papers do not say."`
- **Table Lookup Fidelity**: Numerical benchmarks (e.g. BLEU 28.4 on EN-DE) are verified against canonical document tables via `lookup_table`.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
