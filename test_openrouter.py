"""
Probe OpenRouter / Active LLM Endpoint for The Relay.
Validates the API key in .env and tests streaming connectivity with google/gemma-4-31b-it:free.
"""

import asyncio
import os
import sys
from relay.backends.base import LLMMessage, LLMRequest
from relay.backends.openrouter import OpenRouterBackend, load_env_file


async def test_connection():
    load_env_file()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key == "your_openrouter_api_key_here":
        print("[ERROR] Please paste your valid OPENROUTER_API_KEY into the .env file first.")
        sys.exit(1)

    model = os.environ.get("ACTIVE_MODEL", "google/gemma-4-31b-it:free")
    print(f"Connecting to OpenRouter with model: {model}...")

    backend = OpenRouterBackend()
    req = LLMRequest(
        model=model,
        messages=[
            LLMMessage(role="system", content="You are a helpful research assistant."),
            LLMMessage(role="user", content="Reply with: 'OpenRouter connection successful!'")
        ],
        max_tokens=30,
        stream=True
    )

    print("Streaming tokens: ", end="", flush=True)
    async for chunk in backend.stream(req):
        if chunk.delta_text:
            print(chunk.delta_text, end="", flush=True)
    print("\n\n[SUCCESS] OpenRouter streaming backend is working properly!")


if __name__ == "__main__":
    asyncio.run(test_connection())

