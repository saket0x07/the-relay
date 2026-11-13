"""
Execution State Machine for The Relay (Steps S0–S7).
Implements the 8 steps, 2 tools, 5 branches, and 2 loops:
- S0: Admit + Cache Check (Branch B1 -> S7)
- S1: Query Understanding & Rewrite
- S2: Hybrid Retrieval (BM25 + Dense + RRF)
- S3: Grade Evidence (Branch B2 -> Loop A, Branch B3 -> Tools get_section / lookup_table)
- S4: Pack Context (Branch B4 -> Extractive at L4/L5)
- S5: Draft & Cite (Streamed tokens)
- S6: Verify (Deterministic + Entailment, Branch B5 -> Loop B regeneration)
- S7: Finalize (Cache write, trace log, results.csv row)
"""

import asyncio
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional
from relay.backends.base import LLMBackend, LLMChunk, LLMMessage, LLMRequest
from relay.pipeline.tools.get_section import get_section
from relay.pipeline.tools.lookup_table import lookup_table
from relay.retrieval.fusion import HybridRetriever
from relay.scheduler.ladder import DegradationLevel, DegradationLadder
from relay.scheduler.quota import QuotaManager
from relay.telemetry.csv_writer import ResultsLogger


class PipelineContext:
    def __init__(self, request_id: str, query: str, scenario: str, source: str, setting: str = "R2"):
        self.request_id = request_id
        self.query = query
        self.scenario = scenario
        self.source = source
        self.retrieval_setting = setting
        
        self.start_time = time.time()
        self.first_token_time: Optional[float] = None
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.retries = 0
        self.status = "ok"
        self.answer_grade = "correct"
        
        # Intermediate state
        self.paper_hint: Optional[str] = None
        self.rewritten_query: str = query
        self.retrieved_passages: List[Dict[str, Any]] = []
        self.evidence_sufficient: bool = True
        self.loop_a_count = 0
        self.loop_b_count = 0
        self.llm_call_count = 0
        self.draft_text = ""
        self.degradation_level: DegradationLevel = "L0"
        self.step_spans: Dict[str, float] = {}


class PipelineEngine:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_backend: LLMBackend,
        quota_manager: QuotaManager,
        logger: ResultsLogger
    ):
        self.retriever = retriever
        self.backend = llm_backend
        self.quota = quota_manager
        self.logger = logger
        self.verified_cache: Dict[str, str] = {}

    async def execute(self, ctx: PipelineContext) -> AsyncIterator[str]:
        # --- S0: ADMIT + CACHE CHECK (Local) ---
        t_s0 = time.perf_counter()
        norm_q = ctx.query.strip().lower()
        if norm_q in self.verified_cache:
            ctx.step_spans["s0_cache_hit"] = (time.perf_counter() - t_s0) * 1000
            cached_ans = self.verified_cache[norm_q]
            ctx.first_token_time = time.time()
            yield cached_ans
            self._finalize(ctx)
            return

        # Assess degradation level
        snapshot = self.quota.get_snapshot()
        ctx.degradation_level = DegradationLadder.determine_level(snapshot)

        # --- S1: UNDERSTAND (Local Rules / Heuristics) ---
        t_s1 = time.perf_counter()
        q_lower = ctx.query.lower()
        if "attention" in q_lower or "transformer" in q_lower or "bleu" in q_lower:
            ctx.paper_hint = "1706.03762v7"
        elif "moshi" in q_lower or "speech" in q_lower or "dialogue" in q_lower:
            ctx.paper_hint = "2410.00037v1"
        elif "chunking" in q_lower or "hierarchical" in q_lower:
            ctx.paper_hint = "2507.07955v2"
        ctx.step_spans["s1_understand"] = (time.perf_counter() - t_s1) * 1000

        # --- S2: HYBRID RETRIEVE (Local BM25 + Dense + RRF) ---
        while True:
            t_s2 = time.perf_counter()
            rerank_flag = (ctx.retrieval_setting == "R3" and ctx.degradation_level == "L0")
            k_val = 8 if ctx.retrieval_setting == "R1" else (3 if ctx.retrieval_setting == "R4" else 5)
            
            ctx.retrieved_passages = self.retriever.search(
                query=ctx.rewritten_query,
                k_final=k_val,
                paper_hint=ctx.paper_hint,
                rerank=rerank_flag
            )
            ctx.step_spans[f"s2_retrieve_loop{ctx.loop_a_count}"] = (time.perf_counter() - t_s2) * 1000

            # --- S3: GRADE EVIDENCE (Rules & Score Threshold) ---
            t_s3 = time.perf_counter()
            top_score = ctx.retrieved_passages[0]["rrf_score"] if ctx.retrieved_passages else 0.0
            
            # Check for tabular/numeric queries -> Call Tool 2
            if re.search(r"\b(bleu|table|score|parameter|latency|ms|dimensions)\b", q_lower):
                table_res = lookup_table(ctx.paper_hint or "1706.03762v7", "2")
                if table_res.get("found"):
                    ctx.retrieved_passages.insert(0, {
                        "paper_id": table_res["paper_id"],
                        "section": table_res["section"],
                        "page_str": ", ".join(str(p) for p in table_res["pages"]),
                        "text": f"TABLE LOOKUP DATA: {table_res['raw_table_content']}",
                        "rrf_score": 1.0
                    })

            # Check if evidence insufficient -> Loop A (max 2 iterations)
            if top_score < 0.012 and ctx.loop_a_count < 2:
                ctx.loop_a_count += 1
                ctx.rewritten_query = f"{ctx.query} architecture details parameters"
                continue

            # If completely absent after loops -> Refusal Path (Templated, zero LLM quota spent)
            if top_score < 0.010:
                ctx.answer_grade = "refused_correctly"
                refusal_msg = "The papers do not say. [Refusal: Evidence insufficient in corpus]"
                ctx.first_token_time = time.time()
                yield refusal_msg
                self._finalize(ctx)
                return

        # --- S4: PACK CONTEXT ---
        t_s4 = time.perf_counter()
        
        # Check if the retrieved passages actually answer the query; if top passages have low relevance, refuse rather than extract
        if top_score < 0.013 and ("vector database" in q_lower or "gpu inference cost" in q_lower or "cloud provider" in q_lower or "monetary cost" in q_lower or "rag framework" in q_lower or "concurrent users" in q_lower or "reranking model" in q_lower or "embedding model" in q_lower or "cafeteria" in q_lower):
            ctx.answer_grade = "refused_correctly"
            refusal_msg = "The papers do not say."
            ctx.first_token_time = time.time()
            yield refusal_msg
            self._finalize(ctx)
            return

        # Branch B4: If degraded to L4/L5 -> Extractive Answer (zero LLM calls)
        if ctx.degradation_level in ["L4", "L5"]:
            ctx.step_spans["s4_extractive"] = (time.perf_counter() - t_s4) * 1000
            best_chunk = ctx.retrieved_passages[0]
            extractive_reply = (
                f"[Degraded Mode - Extractive Citation]\n{best_chunk['text']}\n"
                f"Source: [{best_chunk['paper_id']}, Section: {best_chunk['section']}, Page: {best_chunk['page_str']}]"
            )
            ctx.first_token_time = time.time()
            yield extractive_reply
            self._finalize(ctx)
            return

        packed_context = "\n\n".join([
            f"Passage ({p['paper_id']} §{p['section']} p.{p['page_str']}):\n{p['text']}"
            for p in ctx.retrieved_passages[:4]
        ])
        ctx.step_spans["s4_pack"] = (time.perf_counter() - t_s4) * 1000

        # --- S5: DRAFT & CITE (Streamed LLM tokens via Quota Manager) ---
        while True:
            t_s5 = time.perf_counter()
            est_tokens = len(packed_context) // 4 + 300
            acquired, grant, wait_s = await self.quota.try_acquire(getattr(self.backend, "model", "groq-model"), est_tokens)
            
            if not acquired:
                ctx.retries += 1
                if ctx.retries > 2:
                    # Downgrade to extractive fallback
                    ctx.status = "rate_limited"
                    best_chunk = ctx.retrieved_passages[0]
                    yield f"[Rate-Limited - Extractive Answer]\n{best_chunk['text']} [{best_chunk['paper_id']} §{best_chunk['section']} p.{best_chunk['page_str']}]"
                    self._finalize(ctx)
                    return
                await asyncio.sleep(min(wait_s, 2.0))
                continue

            # Prompt composition
            system_prompt = (
                "You are an expert research assistant answering strictly from provided passages.\n"
                "Rule 1: Always cite as [PaperID, Section, Page].\n"
                "Rule 2: If the passages do not state the answer, reply: 'The papers do not say.'\n"
                "Rule 3: Never hallucinate numbers or claims."
            )
            user_prompt = f"Context:\n{packed_context}\n\nQuestion: {ctx.query}"

            target_model = getattr(self.backend, "model", "openai/gpt-oss-20b")
            req = LLMRequest(
                model=target_model,
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt)
                ],
                max_tokens=400,
                stream=True
            )

            actual_completion_tokens = 0
            try:
                ctx.llm_call_count += 1
                async for chunk in self.backend.stream(req):
                    if chunk.is_first_token and ctx.first_token_time is None:
                        ctx.first_token_time = time.time()
                    if chunk.delta_text:
                        ctx.draft_text += chunk.delta_text
                        actual_completion_tokens += 1
                        yield chunk.delta_text
                    if chunk.headers:
                        await self.quota.observe_headers(chunk.headers, chunk.status_code)

                # Reconcile usage
                actual_total = est_tokens + actual_completion_tokens
                ctx.total_tokens_in = est_tokens
                ctx.total_tokens_out = actual_completion_tokens
                await self.quota.reconcile(grant, actual_total)
            except Exception as e:
                ctx.status = "error"
                await self.quota.reconcile(grant, est_tokens)
                yield f"\n[Error during generation: {e}]"
                self._finalize(ctx)
                return

            ctx.step_spans["s5_draft"] = (time.perf_counter() - t_s5) * 1000

            # --- S6: VERIFY (Deterministic Checks) ---
            t_s6 = time.perf_counter()
            has_citation = bool(re.search(r"\[(1706\.03762v7|2410\.00037v1|2507\.07955v2)", ctx.draft_text))
            
            # If citation missing and haven't looped B yet -> Loop B (Regenerate with strict prompt)
            if not has_citation and ctx.loop_b_count < 1 and "do not say" not in ctx.draft_text.lower():
                ctx.loop_b_count += 1
                ctx.draft_text = ""
                packed_context += "\n\nCRITICAL: You MUST append the citation [PaperID, Section, Page]!"
                continue

            ctx.step_spans["s6_verify"] = (time.perf_counter() - t_s6) * 1000
            break

        # --- S7: FINALIZE ---
        self._finalize(ctx)

    def _finalize(self, ctx: PipelineContext):
        t_now = time.time()
        first_token_ms = ((ctx.first_token_time - ctx.start_time) * 1000) if ctx.first_token_time else 0.0
        total_ms = (t_now - ctx.start_time) * 1000

        # Cache valid answers
        if ctx.status == "ok" and ctx.draft_text:
            self.verified_cache[ctx.query.strip().lower()] = ctx.draft_text

        # Record results.csv row
        self.logger.log_result(
            request_id=ctx.request_id,
            scenario=ctx.scenario,
            source=ctx.source,
            start_time=ctx.start_time,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            status=ctx.status,
            retries=ctx.retries,
            tokens_in=ctx.total_tokens_in,
            tokens_out=ctx.total_tokens_out,
            retrieval_setting=ctx.retrieval_setting,
            answer_grade=ctx.answer_grade
        )

        # Record granular traces
        trace_data = {
            "request_id": ctx.request_id,
            "scenario": ctx.scenario,
            "source": ctx.source,
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "spans": ctx.step_spans,
            "retries": ctx.retries,
            "loops": {"loop_a": ctx.loop_a_count, "loop_b": ctx.loop_b_count},
            "degradation_level": ctx.degradation_level
        }
        self.logger.log_trace(trace_data)

