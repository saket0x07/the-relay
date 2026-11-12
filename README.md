# The Relay: A Scalable, Quota-Aware RAG Pipeline with Hybrid Search

The Relay is a production-grade, quota-aware Retrieval-Augmented Generation (RAG) system designed to operate under strict API rate limits (RPM, TPM, RPD, and in-flight concurrency) while guaranteeing that no uncited, fabricated, or ungrounded responses are shipped.

## 🚀 Key Features
- **Local Hybrid Retrieval**: BM25 keyword search with technical-token preserving tokenization (`d_model`, hyphenated terms) + FastEmbed dense ONNX vector search (`bge-small-en-v1.5`), fused via Reciprocal Rank Fusion ($k=60$) at zero API token cost.
- **8-Step State Machine**: S0 (Admit/Cache) $\to$ S1 (Understand) $\to$ S2 (Retrieve) $\to$ S3 (Grade) $\to$ S4 (Pack) $\to$ S5 (Draft & Cite) $\to$ S6 (Verify) $\to$ S7 (Finalize).
- **Two Deterministic Tools**: `get_section` for canonical verbatim spans and `lookup_table` for structured table values.
- **6-Stage Graceful Degradation Ladder (L0–L5)**: Rather than dropping users or crashing with 429 errors when quotas bind, the system smoothly degrades from full LLM synthesis down to extractive citations and honest refusals.
- **Pluggable Model Backends**: Supports native Groq Cloud API, OpenAI-compatible endpoints, and a calibrated Stand-in mock service with load-dependent latency distributions.

## 📁 Repository Structure
```
├── config/              # Configuration files (limits, ladder policies, scenarios)
├── corpus/              # arXiv documents ingestion & processed canonical blocks
│   └── manifest.json    # SHA-256 hashes of pinned research papers
├── relay/               # Core Production Engine
│   ├── api/             # FastAPI Gateway with SSE token streaming
│   ├── scheduler/       # Quota Manager, AIMD Concurrency, Priority Queue, Ladder
│   ├── pipeline/        # S0-S7 state machine and tools (get_section, lookup_table)
│   ├── retrieval/       # BM25, Dense ONNX embedding, and RRF Fusion
│   ├── backends/        # GroqBackend, OpenRouterBackend, GenericClient
│   └── telemetry/       # Span telemetry and results logger
├── standin/             # Calibrated Stand-in mock server for heavy load testing
├── loadgen/             # Synthetic load generator (steady, burst, sustained overload)
├── eval/                # 16+ curated questions bank & quality assessment
├── analysis/            # make_tables.py script computing Tables 1, 2a, 2b, 3
└── requirements.txt     # Dependencies
```

## 🛠️ Setup & Installation
```bash
git clone https://github.com/saket0x07/VAANI.git
cd VAANI

# Install dependencies
pip install -r requirements.txt

# Configure your API key in .env
cp .env.example .env   # Add your GROQ_API_KEY
```

## ⚡ Running the System

### 1. Interactive Live Demo
```bash
python run_live_demo.py
```

### 2. Launch FastAPI Streaming Gateway
```bash
python -m uvicorn relay.api.gateway:app --host 127.0.0.1 --port 8000
```
Query via SSE:
```bash
curl -X POST http://127.0.0.1:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the value of d_model in the base Transformer?"}'
```

### 3. Run Benchmark Suite & Stand-in Server
```bash
# In terminal 1:
python -m uvicorn standin.server:app --port 8001

# In terminal 2:
python loadgen/runner.py

# Generate outcome tables:
python analysis/make_tables.py
```
