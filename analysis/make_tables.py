"""
Table Builder for The Relay Assignment Brief.
Computes and formats the required tables directly from results.csv:
- Table 1: Real service limits (from probe / docs)
- Table 2a: Latency per load scenario (p50, p95, p99)
- Table 2b: Outcomes per load scenario (Succeeded %, Rate-limited %, etc.)
- Table 3: Retrieval settings sweep (R1, R2, R3, R4)
"""

import os
import pandas as pd
import numpy as np


def compute_percentiles(series: pd.Series):
    if len(series) == 0:
        return "N/A / N/A / N/A"
    p50 = np.percentile(series, 50)
    p95 = np.percentile(series, 95)
    p99 = np.percentile(series, 99)
    return f"{p50:.1f} / {p95:.1f} / {p99:.1f}"


def generate_tables(csv_path: str = "data/results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print("results.csv is empty.")
        return

    print("=" * 80)
    print("TABLE 2a: LATENCY PER LOAD SCENARIO")
    print("=" * 80)
    print(f"{'Scenario':<25} | {'Source':<10} | {'Requests':<8} | {'First Token p50/p95/p99 (ms)':<30} | {'Total Time p50/p95/p99 (ms)':<30}")
    print("-" * 115)

    scenarios = df["scenario"].unique()
    for sc in scenarios:
        sub = df[df["scenario"] == sc]
        source = sub["source"].iloc[0] if len(sub) > 0 else "N/A"
        req_count = len(sub)
        ft_pct = compute_percentiles(sub["first_token_ms"])
        tot_pct = compute_percentiles(sub["total_ms"])
        print(f"{sc:<25} | {source:<10} | {req_count:<8} | {ft_pct:<30} | {tot_pct:<30}")

    print("\n" + "=" * 80)
    print("TABLE 2b: OUTCOMES PER LOAD SCENARIO")
    print("=" * 80)
    print(f"{'Scenario':<25} | {'Succeeded %':<12} | {'Rate-Limited %':<15} | {'Retries/Req':<12} | {'Tokens/Ans':<12} | {'Cost/1k Ans ($)':<15}")
    print("-" * 105)

    for sc in scenarios:
        sub = df[df["scenario"] == sc]
        total_reqs = len(sub)
        succ = (sub["status"] == "ok").sum() / max(1, total_reqs) * 100.0
        rl = (sub["status"] == "rate_limited").sum() / max(1, total_reqs) * 100.0
        avg_retries = sub["retries"].mean()
        avg_tokens = (sub["tokens_in"] + sub["tokens_out"]).mean()
        # Equivalent paid tier calculation ($0.0001 per 1k tokens)
        cost_per_1k = (avg_tokens * 1000 * 0.0001) / 1000.0

        print(f"{sc:<25} | {succ:<11.1f}% | {rl:<14.1f}% | {avg_retries:<12.2f} | {avg_tokens:<12.1f} | ${cost_per_1k:<14.4f}")

    print("\n" + "=" * 80)
    print("TABLE 3: RETRIEVAL SETTINGS SWEEP")
    print("=" * 80)
    print(f"{'Setting':<10} | {'Questions':<10} | {'Correct & Cited %':<18} | {'Tokens/Ans':<12} | {'First Token p50/p95 (ms)':<25}")
    print("-" * 85)

    settings = df["retrieval_setting"].dropna().unique()
    for st in sorted(settings):
        sub = df[df["retrieval_setting"] == st]
        n_q = len(sub)
        correct_pct = (sub["answer_grade"].isin(["correct", "refused_correctly"])).sum() / max(1, n_q) * 100.0
        avg_tokens = (sub["tokens_in"] + sub["tokens_out"]).mean()
        ft_p50 = np.percentile(sub["first_token_ms"], 50) if len(sub) > 0 else 0
        ft_p95 = np.percentile(sub["first_token_ms"], 95) if len(sub) > 0 else 0

        print(f"{st:<10} | {n_q:<10} | {correct_pct:<17.1f}% | {avg_tokens:<12.1f} | {ft_p50:.1f} / {ft_p95:.1f}")


if __name__ == "__main__":
    generate_tables()

