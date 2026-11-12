"""
Quota Manager with Two-Phase Token Reservation & Header Reconciliation for The Relay.
Enforces RPM, TPM, RPD, in-flight limits, and adapts dynamically to response headers.
"""

import asyncio
import time
from typing import Dict, Mapping, Optional, Tuple


class QuotaGrant:
    def __init__(self, model: str, est_tokens: int, reservation_id: str):
        self.model = model
        self.est_tokens = est_tokens
        self.reservation_id = reservation_id
        self.granted_at = time.time()


class QuotaManager:
    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 6000,
        rpd_limit: int = 14400,
        max_inflight: int = 5
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.max_inflight = max_inflight

        # Token buckets
        self.rpm_tokens = float(rpm_limit)
        self.tpm_tokens = float(tpm_limit)
        self.rpd_tokens = float(rpd_limit)
        self.inflight = 0
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
        self.counter = 0

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed > 0:
            # Refill RPM (tokens per second)
            self.rpm_tokens = min(float(self.rpm_limit), self.rpm_tokens + elapsed * (self.rpm_limit / 60.0))
            # Refill TPM
            self.tpm_tokens = min(float(self.tpm_limit), self.tpm_tokens + elapsed * (self.tpm_limit / 60.0))
            self.last_refill = now

    async def try_acquire(self, model: str, est_tokens: int) -> Tuple[bool, Optional[QuotaGrant], float]:
        async with self.lock:
            self._refill()

            # Check in-flight ceiling
            if self.inflight >= self.max_inflight:
                return False, None, 0.5

            # Check daily quota
            if self.rpd_tokens < 1.0:
                return False, None, 3600.0  # RPD exhausted

            # Check RPM & TPM availability
            if self.rpm_tokens >= 1.0 and self.tpm_tokens >= est_tokens:
                self.rpm_tokens -= 1.0
                self.tpm_tokens -= est_tokens
                self.rpd_tokens -= 1.0
                self.inflight += 1
                self.counter += 1
                grant = QuotaGrant(model, est_tokens, f"grant_{self.counter}")
                return True, grant, 0.0

            # Calculate wait time needed for refill
            rpm_wait = (1.0 - self.rpm_tokens) / (self.rpm_limit / 60.0) if self.rpm_tokens < 1.0 else 0.0
            tpm_wait = (est_tokens - self.tpm_tokens) / (self.tpm_limit / 60.0) if self.tpm_tokens < est_tokens else 0.0
            wait_s = max(rpm_wait, tpm_wait, 0.1)
            return False, None, min(wait_s, 60.0)

    async def reconcile(self, grant: QuotaGrant, actual_tokens: int):
        """Reconcile difference between estimated tokens and actual tokens."""
        async with self.lock:
            self._refill()
            self.inflight = max(0, self.inflight - 1)
            token_diff = grant.est_tokens - actual_tokens
            # If we over-reserved, refund tokens back into the bucket
            self.tpm_tokens = min(float(self.tpm_limit), self.tpm_tokens + token_diff)

    async def observe_headers(self, headers: Mapping[str, str], status_code: int):
        """Update internal limits directly from provider header feedback."""
        async with self.lock:
            for k, v in headers.items():
                k_lower = k.lower()
                try:
                    if k_lower == "x-ratelimit-remaining-tokens":
                        rem = float(v)
                        # Gently reconcile our local bucket with provider's source of truth
                        self.tpm_tokens = min(self.tpm_tokens, rem)
                    elif k_lower == "x-ratelimit-remaining-requests":
                        rem = float(v)
                        self.rpm_tokens = min(self.rpm_tokens, rem)
                    elif k_lower == "x-ratelimit-limit-tokens":
                        self.tpm_limit = int(v)
                    elif k_lower == "x-ratelimit-limit-requests":
                        self.rpm_limit = int(v)
                except ValueError:
                    pass

    def get_snapshot(self) -> Dict[str, float]:
        self._refill()
        return {
            "rpm_remaining": round(self.rpm_tokens, 1),
            "tpm_remaining": round(self.tpm_tokens, 1),
            "rpd_remaining": round(self.rpd_tokens, 1),
            "inflight": self.inflight,
            "rpm_utilization": round(1.0 - (self.rpm_tokens / self.rpm_limit), 3),
            "tpm_utilization": round(1.0 - (self.tpm_tokens / self.tpm_limit), 3)
        }

