# PRD — The Relay: A Scalable, Quota-Aware RAG Pipeline with Hybrid Search

**Stack:** Python 3.11+, `asyncio`, `httpx`, FastAPI (SSE) · **LLM:** Groq free tier (real) + in-house stand-in · **Corpus:** 3 pinned arXiv papers

---

## 1. Problem and Core Insight

Requests arrive in bursts. Each needs several model calls plus retrieval and a tool call. The model provider enforces RPM, TPM, RPD and (implicitly) concurrency limits that we do not control.

**Core insight that drives the design:** the provider quota is a *global, shared, non-scalable resource*. Adding workers does not add capacity. Therefore scalability here means:

1. **Spend the quota deliberately** (budgeting, prioritisation, token-aware admission).
2. **Make everything else elastic** (stateless workers, local retrieval, bounded queues, backpressure).
3. **Degrade by design** when demand exceeds quota, instead of failing.

**Priority order when goals conflict:** correct & cited > nobody turned away > fast (TTFT) > cheap.
Never ship a wrong or uncited answer to save time.

---

## 2. Requirements

### 2.1 Functional
| ID | Requirement |
|---|---|
| F1 | Pipeline of 8 steps (S0–S7), 2 tools, 5 branches, 2 loops (brief minimum: 4 steps, 1 tool, 1 branch/loop) — see §6.1 |
| F2 | **Hybrid retrieval** over the 3 papers (BM25 + dense + fusion, optional rerank) |
| F3 | Every answer cites `paper · section · page`; if the papers don't say, answer "the papers do not say" |
| F4 | Streaming responses (first token ASAP) |
| F5 | Pluggable backend: `GroqBackend` (real) and `StandInBackend` (simulated), same interface |
| F6 | Load generator with ≥3 shapes; every request logged to `results.csv` |

### 2.2 Non-functional (targets are *hypotheses* until measured)
| Area | Target |
|---|---|
| Latency | TTFT p95 ≤ 3 s steady; ≤ 8 s burst (includes queue wait) |
| Correctness | Wrong-answer rate ≤ 5%; zero uncited claims shipped |
| Availability | 100% of accepted requests get a truthful reply (answer, degraded answer, or honest status) |
| Overload | Bounded memory; queue never grows unbounded; p99 degrades gracefully, not exponentially |
| Elasticity | Retrieval + orchestration scale horizontally with no code change; quota state is the only shared state |
| Observability | Per-step timing, queue length, degradation level, token use |

---

## 3. System Architecture

```
                         ┌────────────────────────── CONTROL PLANE ───────────────────────────┐
                         │  Quota Manager  ·  Adaptive Concurrency (AIMD)  ·  Degradation     │
                         │  Controller  ·  Circuit Breaker  ·  Retry Budget                   │
                         └───────────▲───────────────────────────────▲──────────────────────────┘
                                     │ policy / signals              │ headers, 429s, latency
 client ─► API Gateway ─► Admission ─┴─► Priority/Deadline Queue ─► Worker Pool ─► Pipeline Engine
 (SSE)     (FastAPI)     (accept all,        (bounded, per-class)    (N async      (S0→S7, §6.1)
                          assign deadline,                           workers)            │
                          class, budget)                                                 ▼
                                                              ┌──────────────┬───────────────────┬──────────────┐
                                                              │ Retrieval    │ Backend Adapter   │ Tool Runner  │
                                                              │ (local,      │ Groq | Stand-in   │ (timeouts,   │
                                                              │ hybrid)      │ via Quota Manager │ fallbacks)   │
                                                              └──────────────┴───────────────────┴──────────────┘
                                                                                   │
                                        Telemetry (trace JSONL + results.csv) ◄────┘
```

### 3.1 Components and responsibilities

| Component | Responsibility | Scaling property |
|---|---|---|
| **API Gateway** | Accept request, open SSE stream, return immediately with request id and state | Stateless, horizontal |
| **Admission** | Always accepts. Assigns `deadline`, `class` (short / long / loop-suspect), `token_budget`, `max_steps` | Stateless |
| **Queue** | Bounded, deadline-ordered (least slack first), per-class lanes so long/looping requests can't starve short ones | In-proc heap now; Redis sorted set when multi-node |
| **Worker pool** | Runs pipeline per request using `asyncio` tasks; pool size tied to concurrency limit, not CPU | Horizontal |
| **Pipeline engine** | State machine executing steps with per-step budgets, branch and loop logic | Stateless per request |
| **Retrieval service** | Local hybrid search; zero provider tokens | CPU-bound, shardable, replicable |
| **Backend adapter** | Uniform `stream(prompt, model, max_tokens)` returning tokens and usage; all calls pass through Quota Manager | Per-provider |
| **Quota Manager** | Token buckets (RPM, TPM, RPD, TPD), in-flight semaphore, header-driven learning | **The only globally shared state** |
| **Degradation controller** | Picks the service level from utilisation and queue slack | Pure function of metrics |
| **Telemetry** | Timestamps per step → `results.csv` and traces | Append-only |

### 3.2 Request lifecycle (state machine)

```
ACCEPTED → QUEUED → RUNNING(step k) → STREAMING → VERIFIED → DONE
                │          │               │
                │          ├─ retry ───────┘ (budgeted)
                │          └─ degrade ──► EXTRACTIVE / DEFERRED
                └─ expired (deadline) ──► GIVEN_UP (removed before wasting quota)
```

---

## 4. Scalability Design

### 4.1 Capacity model (use Little's Law, then measure)
- `in_flight = arrival_rate × time_in_system`
- `calls_per_request`: 3 LLM calls on the normal path (S1, S5, S6), up to 6 worst case (both loops). Retrieval, tools, grading and packing are local and cost no quota.
- Provider-bound throughput: `max_requests/min ≈ min( RPM_limit / calls_per_request , TPM_limit / tokens_per_request )`
- **Whichever of RPM or TPM binds first sets system capacity.** On a free tier with ~1–3k-token RAG prompts, TPM usually binds before RPM — verify by measurement. This is why *context size* is a scaling lever.
- Worked example (fill with real numbers): if TPM = T and tokens/answer = A, sustainable rate = T / A answers/min. Everything above that is queued, degraded or deferred.

### 4.2 Levers that actually raise effective capacity
| Lever | Effect on quota | Cost |
|---|---|---|
| Smaller context (fewer/smaller chunks, rerank for precision) | ↓ tokens per answer → ↑ answers/min | Slight recall risk |
| Model-tier routing (small model for S1/S5, large for S4) | Uses separate per-model limits | Quality on S4 only |
| Skip LLM where rules suffice (S1 fallback, deterministic verify) | ↓ calls per request | Weaker disambiguation |
| Answer cache (verified answers only) | Zero quota for repeats | Staleness only on corpus change |
| Early shedding of expired requests | Stops spending quota on abandoned users | None |
| Local retrieval | No quota used | CPU |

### 4.3 Backpressure and bounded resources
- Bounded queue (size from `rate × max_wait`), per-class caps, bounded in-flight, bounded streaming buffers.
- When full: **do not reject.** Convert to the cheapest truthful path (extractive answer or honest deferred status).
- Workers pull when quota is available. They never push into the provider and wait on 429s.

### 4.4 Scale-out path (design now, build only the first step)
| Stage | Topology | Shared state |
|---|---|---|
| **v1 (build)** | Single process, `asyncio`, in-memory queue + buckets | none |
| **v2 (design only)** | N gateway/worker processes | Redis: token buckets (atomic Lua), queue, answer cache |
| **v3 (design only)** | Separate retrieval service; sharded index; multiple providers/keys *allowed by terms* | Redis + service discovery |

Interfaces below are written so v1 → v2 is a swap of `QuotaStore` and `QueueStore` implementations, not a rewrite.

---

## 5. Control Plane Spec

### 5.1 Quota Manager
- **Token buckets:** RPM, TPM, RPD, TPD. TPM reserves `tokens_in_est + max_tokens`, then **reconciles with actual usage** on completion (refund or debt).
- **Header-driven truth:** parse `x-ratelimit-*` / `retry-after` from every response; config limits are only priors. Limits can change mid-run, so recompute capacity continuously.
- **Concurrency:** AIMD — +1 slot after K successes; ×0.5 on 429 or timeout; floor and cap configurable.
- **Admission to the provider = all of:** RPM token, TPM tokens, in-flight slot, breaker closed.

```python
class QuotaStore(Protocol):            # swap impl for Redis in v2
    async def try_acquire(self, model: str, est_tokens: int) -> Grant | Denied(wait_s: float)
    async def reconcile(self, grant: Grant, actual_tokens: int) -> None
    async def observe(self, model: str, headers: Mapping, status: int) -> None
    def snapshot(self, model: str) -> QuotaState   # remaining RPM/TPM/RPD, in_flight, utilisation
```

### 5.2 Scheduling
- **Order:** least slack first, where `slack = deadline − now − est_remaining_work`.
- **Per-class lanes** with weighted share, so a looping request cannot starve short ones.
- **Loop-suspect cap:** a request that re-enters retrieval more than 2 times (Loop A), regenerates more than once (Loop B), exceeds **6 LLM calls**, or exceeds its token budget is finalised with the best available answer or an honest refusal.
- **Shed before spend:** a request whose slack is negative and that has not yet used quota is dropped to the cheapest path or marked given-up.

### 5.3 Degradation ladder (weaker answer, never a wrong one)
| Level | Trigger (tune from data) | Action | Protected |
|---|---|---|---|
| L0 | util < 60% | Large model, rerank, k=5 | — |
| L1 | 60–80% | Rerank off, k=4 | Verify |
| L2 | 80–95% or TPM bucket < 25% | Small model everywhere, rule-based S1 | Verify |
| L3 | > 95% or queue wait > threshold | Single retrieval pass, short answer, deterministic verify | Citations, "papers don't say" |
| L4 | Breaker open / RPD exhausted | **Extractive**: top passages with citations, honest note | Honesty |
| L5 | Extractive confidence low | Honest "can't answer well now; resets at HH:MM" | Never guess |

Hysteresis on transitions (enter fast, exit slow) to avoid flapping.

### 5.4 Resilience patterns
| Pattern | Rule |
|---|---|
| **Timeouts** | Per call and per step, derived from measured latency percentiles. Separate first-token timeout and total deadline (catches slow successes) |
| **Retries** | Only 429 / 5xx / connect errors / timeouts. Honour `retry-after`, else exponential backoff with full jitter. Never retry per-day 429 or 4xx. Retries re-enter the scheduler and consume RPM. **Retry budget ≤ ~10–20% of recent traffic.** Skip retry if remaining deadline < expected call time. No blind retry after tokens were streamed |
| **Circuit breaker** | Open on sustained 429/5xx/slow-success rate. Half-open probes. Open state routes to L4 |
| **Bulkheads** | Separate quota partitions for interactive vs eval traffic; the eval harness cannot drain the production budget |
| **Idempotency** | Request id on every step so retries and hedges don't double-count |
| **Hedging** | Off by default: it burns RPM. Document the decision |

---

## 6. Data Plane Spec

### 6.1 Final pipeline (8 steps, 2 tools, 5 branches, 2 loops)

```
 request
    │
 S0  ADMIT + CACHE CHECK (local)
    │──[B1] verified-answer cache hit ───────────────────────────────► S7
    ▼
 S1  UNDERSTAND (small LLM, rules fallback)
    │   intent · rewritten query · sub-questions · paper hints · answerable prior
    ▼
 S2  HYBRID RETRIEVE (local)  ◄──────────────────────────┐
    │   BM25 + dense + RRF (+rerank at L0)               │ LOOP A
    ▼                                                    │ (max 2: expand query,
 S3  GRADE EVIDENCE (rules + score thresholds)           │  split sub-questions,
    │                                                    │  switch setting)
    │──[B2] insufficient ────────────────────────────────┘
    │──[B2] absent (after loops exhausted) ──► REFUSAL PATH (templated, no LLM) ──► S7
    │──[B3] needs more context ──► TOOL 1: get_section(paper, section|page)
    │──[B3] numeric / table question ──► TOOL 2: lookup_table(paper, table_id)
    ▼
 S4  PACK CONTEXT (local)
    │   dedupe · merge adjacent · trim to token budget for current level
    │──[B4] degradation L4/L5 ──► EXTRACTIVE ANSWER (top passages + citations) ──► S7
    ▼
 S5  DRAFT + CITE (large LLM, streamed)  ◄────────────┐
    ▼                                                 │ LOOP B
 S6  VERIFY (deterministic checks + small LLM)        │ (max 1: regenerate
    │                                                 │  with stricter prompt)
    │──[B5] fail ─────────────────────────────────────┘
    │──[B5] fail again ──► downgrade to extractive or refusal ──► S7
    ▼ pass
 S7  FINALIZE (local)
        write verified answer to cache · emit trace · write results.csv row
```

**Brief requirements met:** 8 steps (need ≥4) · 2 tool calls (need ≥1) · 5 branches and 2 loops (need ≥1) · hybrid retrieval step (S2).

| # | Step | Runs | Uses Groq quota? | Fallback when quota is short or it fails |
|---|---|---|---|---|
| S0 | Admit + cache check | Local | No | Skip cache, continue |
| S1 | Understand | Small LLM | **Yes** | Rule-based rewrite and paper hints |
| S2 | Hybrid retrieve | Local | No | BM25 only if embedder fails |
| S3 | Grade evidence | Local rules (LLM only if borderline) | Rarely | Rules decide alone |
| T1/T2 | `get_section`, `lookup_table` | Local | No | Timeout, then use already-retrieved context |
| S4 | Pack context | Local | No | Smaller k / chunks per level |
| S5 | Draft + cite | Large LLM, streamed | **Yes** | Small model, then extractive |
| S6 | Verify | Deterministic, then small LLM | **Yes** (LLM part) | Deterministic checks only |
| S7 | Finalize | Local | No | none needed |

**Branches**
- **B1 cache:** serve a previously *verified* answer, with zero quota.
- **B2 evidence grade (3-way):** sufficient → continue; insufficient → Loop A; absent → templated refusal with no LLM call.
- **B3 tool choice:** a section or page fetch for more context, or a table lookup for numeric questions.
- **B4 degradation gate:** at L4/L5 skip drafting and return an extractive answer.
- **B5 verify result:** pass → finalize; fail → Loop B; fail twice → extractive or refusal.

**Tools**
- `get_section(paper, section|page)` returns exact section or page text from the canonical store.
- `lookup_table(paper, table_id)` returns a table's cells so numeric answers aren't read from scrambled PDF text.
- Both have a 1–2 s timeout and fall back to the context already retrieved.

**Loops and termination:** Loop A runs at most 2 times and Loop B at most 1. Every request also has a **hard cap of 6 LLM calls**, a token budget and a wall-clock deadline. Hitting any limit ends in an honest refusal or extractive answer, never a guess.

**Call count:** the normal path makes 3 LLM calls (S1, S5, S6) and the worst case 5–6. This is the `calls_per_request` figure in the §4.1 capacity math.

**Stream vs verify:** run deterministic checks inline and flush S5 output at sentence boundaries. Run the LLM entailment check in S6 concurrently, and halt or correct on failure. Under load switch to hold-back mode (verify, then stream). Measure the TTFT cost of this choice.

**Example traces**
- *Simple factual:* S0 → S1 → S2 → S3 (sufficient) → S4 → S5 → S6 pass → S7 (3 LLM calls).
- *Numeric (e.g., BLEU score):* S3 triggers `lookup_table`; S6 confirms the number appears in the passage.
- *Vague ("how does chunking work?"):* S3 insufficient → Loop A re-retrieves using the paper hint → continue.
- *Not in the papers:* scores stay low through both loops → B2 absent → templated refusal (no draft or verify calls).
- *Overload (L4):* S1, S5 and S6 are skipped and the answer is extractive.

### 6.2 Hybrid retrieval (scalable by construction)

```
query ─► rewrite ─┬─► BM25 top-k_lex ──┐
                  │                     ├─► RRF fusion ─► [optional cross-encoder rerank] ─► dedupe + merge adjacent ─► pack to token budget
                  └─► dense top-k_den ──┘
                  (paper hint from S1 = soft boost, not hard filter)
```

| Aspect | Decision |
|---|---|
| Lexical | BM25 (rank_bm25 or SQLite FTS5) with a technical tokenizer that keeps `d_model`, hyphenated terms, numbers |
| Dense | Small **local** embedder (BGE-small class via fastembed): costs no quota and works at the limit |
| Index | In-memory NumPy matrix (a few hundred chunks). Interface allows FAISS / sharded service later |
| Fusion | RRF (k≈60): rank-based, no score calibration |
| Rerank | Small local cross-encoder, enabled only at L0 |
| Chunk metadata | `paper, section, page, chunk_type (body/table/caption/refs)`. References are down-ranked or excluded |
| Ingestion | Pinned PDF versions + SHA-256 manifest. Canonical sentence store → chunk variants built from it. Index build is offline, versioned and hot-swappable |
| **Quota-aware** | Under pressure, shrink *context* (smaller k / pieces, no rerank) rather than skip retrieval |
| Scale path | Embedding cache; precomputed indices per setting; retrieval as a separate replicable service in v3 |

```python
class Retriever(Protocol):
    async def search(self, q: Query, setting: RetrievalSetting, budget_tokens: int) -> list[Passage]
class RetrievalSetting(BaseModel):
    chunk_tokens: int; k_final: int; k_candidates: int; rerank: bool; fusion: Literal["rrf","weighted"]
```

### 6.3 Retrieval settings to sweep (≥3)
| ID | Piece | Fetched | Rerank |
|---|---|---|---|
| R1 | ~200 tok | 8 | no |
| R2 | ~400 tok (para-aware, 15% overlap) | 5 | no |
| R3 | ~400 tok | 5 from 20 candidates | yes |
| R4 | ~800 tok | 3 | no |

Pick one default by the measured trade-off (correct+cited %, tokens/answer, TTFT). Define the near-limit setting separately (e.g., R2 with smaller k).

### 6.4 Backend adapter
```python
class LLMBackend(Protocol):
    def stream(self, req: LLMRequest) -> AsyncIterator[Chunk]   # yields tokens; final chunk has usage
# Raises typed errors: RateLimited(retry_after, scope=minute|day), Timeout, ServerError, SlowStall
```
Both `GroqBackend` and `StandInBackend` implement this. The pipeline never knows which one it is using.

---

## 7. Stand-In Service (load-test infrastructure)

OpenAI/Groq-compatible `/chat/completions` with SSE. All behaviours are config-driven:

| Behaviour | Mechanism |
|---|---|
| Latency | TTFT and inter-token delay sampled from distributions **fitted to real probe data** (mixture with heavy tail) |
| Failures | Rates for 5xx, resets, malformed streams, mid-stream stalls |
| Limits | Enforces RPM / TPM / RPD / in-flight caps; returns 429 with Groq-shaped headers, body, `retry-after` |
| Load-dependent slowdown | `latency = base × (1 + α·(inflight/capacity)^β)` |
| Slow success | Tail mode: correct but very late |
| Streaming | Chunked tokens with realistic gaps |
| Scripted events | Halve RPM mid-run; zero RPD at time T; reset on trigger |
| Content | Templated. It simulates timing and limits only, so **answer quality is measured on the real service only** |

**Calibration:** fit from real probes, then replay the same small workload on both and compare p50/p95. Report the error. Known gaps: real tail latency is correlated across requests; real contention comes from other tenants; real limits are opaque and can change.

---

## 8. Performance Test Plan

### 8.1 Real-service probe (budgeted)
Write a quota budget first, because RPD is shared by everything. Measure: latency spread by prompt/output size (~50–100 calls), RPM hit and recovery time, TPM hit, in-flight ceiling by ramping concurrency, RPD/TPD from docs and headers (do not exhaust it). Record each limit with value, method and confidence.

### 8.2 Load scenarios
| # | Scenario | Backend | Shape |
|---|---|---|---|
| 1 | Steady | Real (small) + stand-in | Poisson at ~50% of measured capacity |
| 2 | Burst | Stand-in | 10× step for ~30 s |
| 3 | Sustained overload | Stand-in | 2–3× capacity for minutes |
| 4 | Per-minute limit halved | Stand-in | Mid-run RPM/TPM cut 50% (tests adaptation) |
| 5 | Daily budget exhausted | Stand-in (+ real 429 shape) | RPD → 0 mid-run; ladder reaches L4/L5 |
| 6 | Retry-storm ablation | Stand-in | Naive retry vs. budgeted retry |

Mixed request types (e.g., 70% short / 20% long / 10% loop-suspect), fixed seeds.

### 8.3 Metrics (one definition everywhere)
- **TTFT** and **total** measured from request *arrival* (includes queue wait).
- p50/p95/p99 over `status=ok`; report N (p99 needs ≥ ~100 samples).
- Rate-limited %, given-up %, retries/request, peak queue length, tokens per answer, cost per 1,000 answers (equivalent paid rate).
- Quality: ≥15 self-written questions (target ~25: single-paper, numeric/table, multi-hop, ambiguous term, ~6 unanswerable incl. traps). Grades: `correct`, `wrong`, `refused_correctly`, `refused_wrongly`.

### 8.4 Raw output contract
`results.csv`, one row per request:
`request_id, scenario, source, start_time, first_token_ms, total_ms, status, retries, tokens_in, tokens_out, retrieval_setting, answer_grade`
A single script `make_tables.py` computes every table from it. Trace JSONL (per-step enqueue/start/first-byte/end, retries, level, queue length) backs the "where the time goes" analysis.

---

## 9. Library Pitfalls the Retriever Must Handle

| Pitfall | Example failing query | Mitigation |
|---|---|---|
| **Same term, different meaning across papers** (chunking, attention, hierarchical, latency) | "How does chunking work?" (Dynamic Chunking ≠ RAG text splitting) | Paper-aware soft boost, disambiguation in S1, paper shown in every citation |
| **PDF damage to tables, equations, columns** | "What BLEU did the big model get on EN-DE?" | Table-aware chunk type, `get_section` tool, deterministic numeric check |
| **Lexical bait and unanswerables** (references, abstract/intro/conclusion duplicates, plausible-sounding gaps) | "Which paper reports X?" where X is absent | Exclude/down-rank references, dedupe, **evidence-sufficiency threshold** and refusal path |

---

## 10. Build Plan (milestones with acceptance criteria)

| Phase | Build | Done when |
|---|---|---|
| **M1 Foundation** | Ingest + manifest, chunk variants, BM25 + dense + RRF, `LLMBackend` + Groq client, telemetry skeleton | A query returns cited passages; one real call logged to CSV |
| **M2 Pipeline** | S0–S7 with all 5 branches, both loops, both tools; streaming; verify | End-to-end answer with citations; refusal works on unanswerable Qs |
| **M3 Control plane** | Quota Manager (buckets, AIMD, header learning), deadline queue, retries, breaker, ladder | Under a synthetic limit, no unbounded queue; ladder transitions visible in traces |
| **M4 Stand-in + calibration** | Stand-in server, probe real service, fit, validate | Stand-in p50/p95 within ~15–20% of real |
| **M5 Load + eval** | Scenarios 1–6, retrieval sweep, quality eval, `make_tables.py` | Tables regenerate from `results.csv`; retry-storm shown |
| **M6 Hardening** | Tune thresholds from data, cut dead features, rehearse changing a limit, threshold or retrieval setting live | Every knob explainable and changeable in config |

### 10.1 Repo layout
```
relay/
  config/        limits.yaml · standin.yaml · scenarios.yaml · retrieval.yaml · ladder.yaml
  relay/
    api/         gateway.py · admission.py
    scheduler/   quota.py · aimd.py · queue.py · ladder.py · breaker.py · retry.py
    pipeline/    engine.py · admit_cache.py · understand.py · grade.py · tools.py (get_section, lookup_table) · pack.py · draft.py · verify.py · extractive.py · finalize.py
    retrieval/   ingest.py · chunkers.py · bm25.py · dense.py · fusion.py · rerank.py · pack.py
    backends/    base.py · groq.py · standin_client.py
    telemetry/   trace.py · csv_writer.py
  standin/       server.py · calibrate.py
  loadgen/       arrivals.py · scenarios.py · run.py
  eval/          questions.yaml · grade.py · sweep.py
  analysis/      make_tables.py
```

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Provider limits differ from docs or change mid-run | Header-driven learning; priors only from config |
| Shared RPD burned early | Quota budget plan; bulkhead eval traffic; small model for sweeps; stand-in for load |
| Stand-in unrepresentative | Validation run; state error and known gaps |
| Streaming exposes a wrong answer before verify finishes | Sentence-boundary flush after deterministic checks; hold-back mode under load |
| Over-building v2/v3 | Interfaces only; build v1 fully and be able to explain every knob |
| PDF parsing noise | Table-aware chunks, tool, numeric verification |

## 12. Brief Compliance (one-line checklist)
Own Groq key, never committed · no extra accounts or limit tricks · pinned paper versions · ≥15 own questions · `results.csv` columns exactly as specified · only submit what can be defended and modified live.

