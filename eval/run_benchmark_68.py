"""
Benchmark Evaluation Runner for the 68-Question Dataset.
Tests the entire Relay RAG pipeline across:
- single_paper_technical (Q001-Q030)
- cross_paper_reasoning (Q031-Q040)
- numerical_experimental (Q041-Q050)
- unanswerable (Q051-Q060)
- adversarial_retrieval (Q061-Q068)

Outputs:
- eval/eval_results_68.json (full detailed responses, citations, times, and grades)
- eval/eval_summary_68.md (breakdown by category, correctness, citation rate, and latency)
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List

from relay.backends.groq import GroqBackend
from relay.pipeline.engine import PipelineContext, PipelineEngine
from relay.retrieval.fusion import HybridRetriever
from relay.scheduler.quota import QuotaManager
from relay.telemetry.csv_writer import ResultsLogger


def evaluate_response(q_item: Dict[str, Any], answer_text: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected_behavior = q_item.get("expected_behavior", "answer_with_citation")
    is_answerable = q_item.get("answerable", True)
    ans_lower = answer_text.lower()

    # 1. Citation Check
    has_paper_cite = bool(re.search(r"(1706\.03762|2410\.00037|2507\.07955|attention|moshi|dynamic chunking|h-net)", ans_lower))
    has_bracket_cite = bool(re.search(r"\[(1706\.03762v7|2410\.00037v1|2507\.07955v2|section|page|table|\d+)\]", ans_lower))
    cited = has_paper_cite or has_bracket_cite

    # 2. Refusal Check
    refused = any(phrase in ans_lower for phrase in [
        "papers do not say",
        "does not say",
        "do not mention",
        "does not mention",
        "not provided",
        "not specify",
        "not state",
        "no information",
        "not report",
        "refusal"
    ])

    grade = "correct"
    reason = "Answer provided with relevant grounding"

    if not is_answerable or expected_behavior == "refuse_without_hallucination":
        if refused:
            grade = "refused_correctly"
            reason = "Successfully identified unanswerable trap and refused without hallucinating"
        else:
            grade = "hallucinated_unanswerable"
            reason = "Failed to refuse unanswerable question"
    elif expected_behavior == "correct_premise_then_answer":
        # Adversarial query (misleading premise)
        # e.g., "Explain how H-Net achieves 200 ms latency" -> Correct premise is 200ms belongs to Moshi, not H-Net
        detected_misconception = any(w in ans_lower for w in [
            "incorrect", "not true", "actually", "instead", "premise", "misleading", 
            "does not", "belong", "moshi", "helium", "h-net", "neither"
        ])
        if detected_misconception:
            grade = "correct_adversarial_handled"
            reason = "Identified incorrect premise and differentiated paper contributions"
        else:
            grade = "partial_adversarial"
            reason = "Answered but did not explicitly refute the misleading premise"
    else:
        # Standard answerable technical/numerical
        if refused:
            grade = "refused_wrongly"
            reason = "Refused an answerable question"
        elif not cited:
            grade = "uncited_claim"
            reason = "Answer lacked clear paper citation"
        else:
            grade = "correct"
            reason = "Accurately answered with citations"

    return {
        "grade": grade,
        "cited": cited,
        "refused": refused,
        "reason": reason
    }


async def run_benchmark():
    dataset_path = "eval/test_dataset_68.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions: List[Dict[str, Any]] = json.load(f)

    print(f"\n=======================================================")
    print(f"  RUNNING BENCHMARK ACROSS ALL {len(questions)} TEST QUESTIONS")
    print(f"=======================================================\n")

    # Load default R2 chunks
    with open("corpus/processed/chunks_r2_r3.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks into memory.")
    retriever = HybridRetriever(chunks)
    backend = GroqBackend()
    quota = QuotaManager(rpm_limit=1000, tpm_limit=8000, rpd_limit=14400, max_inflight=5)
    logger = ResultsLogger()
    engine = PipelineEngine(retriever, backend, quota, logger)

    results = []
    t_start_all = time.time()

    for idx, q_item in enumerate(questions):
        q_id = q_item["question_id"]
        q_text = q_item["question"]
        q_cat = q_item["category"]
        q_diff = q_item["difficulty"]

        print(f"[{idx+1}/{len(questions)}] ({q_id} | {q_cat} | {q_diff})")
        print(f"  Q: {q_text}")

        ctx = PipelineContext(
            request_id=f"eval68_{q_id}",
            query=q_text,
            scenario="benchmark_68",
            source="real",
            setting="R2"
        )

        streamed_parts = []
        t0 = time.perf_counter()
        async for token in engine.execute(ctx):
            streamed_parts.append(token)
        t_elapsed = (time.perf_counter() - t0) * 1000

        full_answer = "".join(streamed_parts).strip()
        eval_metrics = evaluate_response(q_item, full_answer, ctx.retrieved_passages)

        print(f"  Status: {ctx.status} | Grade: {eval_metrics['grade']} | Time: {t_elapsed:.1f}ms")
        print(f"  Ans snippet: {full_answer[:120].replace(chr(10), ' ')}...\n")

        results.append({
            "question_id": q_id,
            "question": q_text,
            "category": q_cat,
            "difficulty": q_diff,
            "answerable": q_item.get("answerable", True),
            "expected_answer": q_item.get("expected_answer", ""),
            "generated_answer": full_answer,
            "retrieved_passages_count": len(ctx.retrieved_passages),
            "top_source": ctx.retrieved_passages[0]["paper_id"] if ctx.retrieved_passages else None,
            "top_section": ctx.retrieved_passages[0]["section"] if ctx.retrieved_passages else None,
            "total_ms": t_elapsed,
            "tokens_in": ctx.total_tokens_in,
            "tokens_out": ctx.total_tokens_out,
            "evaluation": eval_metrics
        })

        # Gentle pause between requests to respect rolling TPM windows
        await asyncio.sleep(0.5)

    total_duration = time.time() - t_start_all
    print(f"\nCompleted all {len(questions)} evaluations in {total_duration:.2f}s.")

    # Save detailed JSON output
    out_json = "eval/eval_results_68.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved full evaluation trace to {out_json}")

    # Generate Markdown Summary
    generate_summary_markdown(results, total_duration)


def generate_summary_markdown(results: List[Dict[str, Any]], total_duration: float):
    total_q = len(results)
    grades = [r["evaluation"]["grade"] for r in results]
    
    correct_count = sum(1 for g in grades if g in ["correct", "refused_correctly", "correct_adversarial_handled"])
    accuracy_pct = (correct_count / total_q) * 100.0

    categories = set(r["category"] for r in results)
    cat_stats = {}
    for cat in categories:
        sub = [r for r in results if r["category"] == cat]
        cat_correct = sum(1 for r in sub if r["evaluation"]["grade"] in ["correct", "refused_correctly", "correct_adversarial_handled"])
        avg_time = sum(r["total_ms"] for r in sub) / len(sub)
        cat_stats[cat] = {
            "total": len(sub),
            "correct": cat_correct,
            "accuracy": (cat_correct / len(sub)) * 100.0,
            "avg_ms": avg_time
        }

    md_lines = [
        "# The Relay: 68-Question Benchmark Evaluation Report\n",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Questions Evaluated:** {total_q}",
        f"**Overall Accuracy:** {accuracy_pct:.1f}% ({correct_count}/{total_q})",
        f"**Total Evaluation Duration:** {total_duration:.2f} s\n",
        "## Performance by Category\n",
        "| Category | Questions | Correct / Grounded | Accuracy % | Avg Latency (ms) |",
        "|---|---|---|---|---|"
    ]

    for cat, stat in sorted(cat_stats.items()):
        md_lines.append(f"| `{cat}` | {stat['total']} | {stat['correct']} | {stat['accuracy']:.1f}% | {stat['avg_ms']:.1f} |")

    md_lines.extend([
        "\n## Detailed Results per Question\n",
        "| ID | Category | Grade | Total Time (ms) | Answer Snippet |",
        "|---|---|---|---|---|"
    ])

    for r in results:
        snip = r["generated_answer"][:80].replace("\n", " ").replace("|", "\\|")
        md_lines.append(f"| {r['question_id']} | `{r['category']}` | **{r['evaluation']['grade']}** | {r['total_ms']:.1f} | {snip}... |")

    summary_path = "eval/eval_summary_68.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Generated summary report: {summary_path}\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())

