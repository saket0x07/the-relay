"""
Adaptive Concurrency Control (AIMD) & Degradation Ladder for The Relay.
Implements:
- AIMD: Additive Increase (+1 slot on success streak), Multiplicative Decrease (x0.5 on 429/timeout)
- Ladder: L0 (Peak) -> L1 -> L2 -> L3 -> L4 (Extractive) -> L5 (Refusal)
"""

import math
from typing import Dict, Literal

DegradationLevel = Literal["L0", "L1", "L2", "L3", "L4", "L5"]


class AIMDController:
    def __init__(self, initial_slots: int = 2, min_slots: int = 1, max_slots: int = 8, success_threshold: int = 8):
        self.slots = initial_slots
        self.min_slots = min_slots
        self.max_slots = max_slots
        self.success_threshold = success_threshold
        self.consecutive_successes = 0

    def on_success(self):
        self.consecutive_successes += 1
        if self.consecutive_successes >= self.success_threshold:
            if self.slots < self.max_slots:
                self.slots += 1
            self.consecutive_successes = 0

    def on_failure_or_rate_limit(self):
        self.consecutive_successes = 0
        self.slots = max(self.min_slots, math.floor(self.slots * 0.5))


class DegradationLadder:
    @staticmethod
    def determine_level(
        quota_snapshot: Dict[str, float],
        circuit_breaker_open: bool = False,
        queue_wait_ms: float = 0.0
    ) -> DegradationLevel:
        # L4 / L5: Circuit breaker open or daily budget depleted
        if circuit_breaker_open or quota_snapshot.get("rpd_remaining", 1) <= 0:
            return "L4"

        rpm_util = quota_snapshot.get("rpm_utilization", 0.0)
        tpm_util = quota_snapshot.get("tpm_utilization", 0.0)
        max_util = max(rpm_util, tpm_util)

        # L3: High overload or excessive queue backlog
        if max_util >= 0.95 or queue_wait_ms > 4000.0:
            return "L3"

        # L2: Heavy load
        if max_util >= 0.80 or quota_snapshot.get("tpm_remaining", 1000) < 1500:
            return "L2"

        # L1: Moderate load
        if max_util >= 0.60:
            return "L1"

        # L0: Normal / quiet conditions
        return "L0"

