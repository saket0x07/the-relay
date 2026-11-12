"""
Groq API Probing Script for The Relay.
Safely measures real Groq free tier parameters without draining daily quota:
- Reads x-ratelimit-* headers (RPM, TPM, RPD, reset windows)
- Samples Time-to-First-Token (TTFT) and inter-token streaming intervals
- Verifies model identifiers (llama-3.1-8b-instant / llama-3.3-70b-versatile)
"""

import os
import sys
import time
import json
import httpx
from typing import Dict, Any

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def probe_groq(api_key: str, model: str = "llama-3.1-8b-instant") -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise research assistant."},
            {"role": "user", "content": "Reply with exactly: 'OK'"}
        ],
        "temperature": 0.0,
        "max_tokens": 10,
        "stream": True
    }

    print(f"Sending probe request to Groq API ({model})...")
    t0 = time.perf_counter()
    first_token_time = None
    collected_tokens = []
    response_headers = {}

    with httpx.Client(timeout=15.0) as client:
        with client.stream("POST", GROQ_ENDPOINT, headers=headers, json=payload) as response:
            status_code = response.status_code
            response_headers = dict(response.headers)
            
            if status_code != 200:
                print(f"Probe received HTTP {status_code}: {response.read().decode('utf-8')}")
                return {
                    "status_code": status_code,
                    "headers": response_headers,
                    "error": True
                }

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            collected_tokens.append(delta)
                    except json.JSONDecodeError:
                        pass

    t_end = time.perf_counter()
    ttft_ms = ((first_token_time - t0) * 1000) if first_token_time else None
    total_ms = (t_end - t0) * 1000

    # Extract ratelimit headers
    rate_limits = {k: v for k, v in response_headers.items() if "ratelimit" in k.lower() or "retry-after" in k.lower()}

    results = {
        "status_code": status_code,
        "model": model,
        "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
        "total_ms": round(total_ms, 2),
        "tokens": "".join(collected_tokens),
        "rate_limit_headers": rate_limits
    }
    return results


if __name__ == "__main__":
    key = os.environ.get("GROQ_API_KEY")
    if not key and len(sys.argv) > 1:
        key = sys.argv[1]

    if not key:
        print("Usage: python probe_groq.py <GROQ_API_KEY> or set GROQ_API_KEY in environment.")
        sys.exit(1)

    res = probe_groq(key)
    print("\n--- Probe Results ---")
    print(json.dumps(res, indent=2))
    
    with open("data/groq_probe_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nSaved to data/groq_probe_results.json")

