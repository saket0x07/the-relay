"""
HTTPX Streaming Client for Groq API and Stand-In Service for The Relay.
Supports both endpoints with transparent rate-limit capture and error wrapping.
"""

import json
import time
from typing import AsyncIterator, Dict, Optional
import httpx
from relay.backends.base import (
    LLMBackend,
    LLMChunk,
    LLMRequest,
    RateLimitError,
    TimeoutError,
    TokenUsage
)


class GenericStreamingClient(LLMBackend):
    def __init__(self, endpoint_url: str, api_key: Optional[str] = None):
        self.endpoint_url = endpoint_url
        self.api_key = api_key or "standin-key"

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True
        }

        first_token_emitted = False

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                async with client.stream("POST", self.endpoint_url, headers=headers, json=payload) as response:
                    status_code = response.status_code
                    resp_headers = dict(response.headers)

                    if status_code == 429:
                        retry_after = float(resp_headers.get("retry-after", 2.0))
                        raise RateLimitError(retry_after=retry_after, message="429 Rate Limited")

                    if status_code != 200:
                        body = await response.aread()
                        raise RuntimeError(f"HTTP {status_code}: {body.decode('utf-8')}")

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                yield LLMChunk(
                                    delta_text="",
                                    is_first_token=False,
                                    is_done=True,
                                    headers=resp_headers,
                                    status_code=status_code
                                )
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                usage_dict = chunk.get("usage")
                                usage = None
                                if usage_dict:
                                    usage = TokenUsage(
                                        prompt_tokens=usage_dict.get("prompt_tokens", 0),
                                        completion_tokens=usage_dict.get("completion_tokens", 0),
                                        total_tokens=usage_dict.get("total_tokens", 0)
                                    )

                                is_first = False
                                if delta and not first_token_emitted:
                                    first_token_emitted = True
                                    is_first = True

                                yield LLMChunk(
                                    delta_text=delta,
                                    is_first_token=is_first,
                                    is_done=False,
                                    usage=usage,
                                    headers=resp_headers,
                                    status_code=status_code
                                )
                            except json.JSONDecodeError:
                                pass
            except httpx.TimeoutException:
                raise TimeoutError("LLM call timed out.")

