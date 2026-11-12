"""
Unified LLM Backend Adapter for The Relay.
Supports OpenRouter (defaulting to google/gemma-4-31b-it:free),
Groq (for final deployment), and local Stand-In service.
Reads configuration from environment / .env file.
"""

import json
import os
import re
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


def load_env_file(env_path: str = ".env"):
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


class OpenRouterBackend(LLMBackend):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        endpoint_url: Optional[str] = None
    ):
        load_env_file()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("ACTIVE_MODEL", "google/gemma-4-31b-it:free")
        self.endpoint_url = endpoint_url or os.environ.get(
            "ACTIVE_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions"
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/vaani-relay",
            "X-Title": "The Relay RAG System",
            "Content-Type": "application/json"
        }
        
        target_model = request.model if request.model else self.model
        # Use our active model if request specifies a generic or default model
        if "llama" in target_model.lower() and "openrouter" in self.endpoint_url:
            target_model = self.model

        # Fallback list if specific free model is temporarily rate-limited upstream
        candidate_models = [target_model, "liquid/lfm-2.5-2.6b:free", "nvidia/nemotron-3.5-lightning:free"]

        first_token_emitted = False
        last_error = None

        for model_choice in candidate_models:
            payload = {
                "model": model_choice,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True
            }

            async with httpx.AsyncClient(timeout=45.0) as client:
                try:
                    async with client.stream("POST", self.endpoint_url, headers=headers, json=payload) as response:
                        status_code = response.status_code
                        resp_headers = dict(response.headers)

                        if status_code == 429:
                            last_error = RateLimitError(retry_after=1.5, message=f"429 from {model_choice}")
                            continue  # Try next candidate free model

                        if status_code != 200:
                            body = await response.aread()
                            last_error = RuntimeError(f"HTTP {status_code}: {body.decode('utf-8')}")
                            continue

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
                        return  # Successfully streamed
                except httpx.TimeoutException:
                    last_error = TimeoutError("LLM streaming call timed out.")
                    continue

        if last_error:
            raise last_error

