"""
Groq API Client Adapter for The Relay.
Directly communicates with Groq Cloud endpoint: https://api.groq.com/openai/v1/chat/completions
Extracts x-ratelimit-* headers, stream tokens, handles reasoning/content deltas,
and maps 429 rate limit errors to typed exceptions.
"""

import json
import os
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
from relay.backends.openrouter import load_env_file


class GroqBackend(LLMBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        load_env_file()
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self.endpoint_url = endpoint_url or os.environ.get(
            "GROQ_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # If caller passed a generic name, default to active supported Groq model
        target_model = self.model
        if request.model and request.model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            target_model = request.model

        msgs = []
        for m in request.messages:
            if isinstance(m, dict):
                msgs.append({"role": m["role"], "content": m["content"]})
            else:
                msgs.append({"role": m.role, "content": m.content})

        payload = {
            "model": target_model,
            "messages": msgs,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True
        }

        first_token_emitted = False

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                async with client.stream("POST", self.endpoint_url, headers=headers, json=payload) as response:
                    status_code = response.status_code
                    resp_headers = dict(response.headers)

                    if status_code == 429:
                        retry_after = float(resp_headers.get("retry-after", 2.0))
                        raise RateLimitError(retry_after=retry_after, message="429 Rate Limited from Groq")

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
                                choices = chunk.get("choices", [{}])
                                delta = choices[0].get("delta", {}) if choices else {}
                                content = delta.get("content", "")
                                
                                usage_dict = chunk.get("usage")
                                usage = None
                                if usage_dict:
                                    usage = TokenUsage(
                                        prompt_tokens=usage_dict.get("prompt_tokens", 0),
                                        completion_tokens=usage_dict.get("completion_tokens", 0),
                                        total_tokens=usage_dict.get("total_tokens", 0)
                                    )

                                is_first = False
                                if content and not first_token_emitted:
                                    first_token_emitted = True
                                    is_first = True

                                yield LLMChunk(
                                    delta_text=content,
                                    is_first_token=is_first,
                                    is_done=False,
                                    usage=usage,
                                    headers=resp_headers,
                                    status_code=status_code
                                )
                            except json.JSONDecodeError:
                                pass
            except httpx.TimeoutException:
                raise TimeoutError("Groq streaming call timed out.")
