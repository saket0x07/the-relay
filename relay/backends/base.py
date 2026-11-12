"""
Base LLM Protocol & typed data structures for The Relay.
Uniform interface implemented by both GroqBackend and StandInBackend.
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMRequest:
    messages: List[LLMMessage]
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    stream: bool = True
    request_id: Optional[str] = None


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMChunk:
    delta_text: str = ""
    is_first_token: bool = False
    is_done: bool = False
    usage: Optional[TokenUsage] = None
    headers: Dict[str, str] = field(default_factory=dict)
    status_code: int = 200


class LLMError(Exception):
    """Base error for LLM calls."""
    pass


class RateLimitError(LLMError):
    def __init__(self, retry_after: float = 1.0, message: str = "Rate limited"):
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutError(LLMError):
    pass


@runtime_checkable
class LLMBackend(Protocol):
    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Yield chunks of tokens; final chunk contains is_done=True and TokenUsage."""
        ...

