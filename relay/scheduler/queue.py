"""
Priority Deadline Queue (Least Slack First) with Class Isolation for The Relay.
Prevents looping or complex multi-hop requests from starving short factual queries.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple


class QueuedItem:
    def __init__(
        self,
        request_id: str,
        query: str,
        req_class: str,          # "short", "long", "loop-suspect"
        deadline_ts: float,      # Wall clock deadline
        est_work_s: float = 1.5,
        payload: Any = None
    ):
        self.request_id = request_id
        self.query = query
        self.req_class = req_class
        self.deadline_ts = deadline_ts
        self.est_work_s = est_work_s
        self.payload = payload
        self.enqueued_ts = time.time()

    @property
    def slack(self) -> float:
        """Slack = deadline - now - estimated_remaining_work"""
        return self.deadline_ts - time.time() - self.est_work_s


class PriorityDeadlineQueue:
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._lock = asyncio.Lock()
        self._queue: List[QueuedItem] = []
        self._has_items = asyncio.Event()

    async def push(self, item: QueuedItem) -> bool:
        async with self._lock:
            if len(self._queue) >= self.max_size:
                return False  # Queue saturated
            self._queue.append(item)
            # Least Slack First sorting
            self._queue.sort(key=lambda x: x.slack)
            self._has_items.set()
            return True

    async def pop(self, timeout: float = 1.0) -> Optional[QueuedItem]:
        start = time.time()
        while time.time() - start < timeout:
            async with self._lock:
                # Discard expired items whose slack is negative before wasting quota
                now = time.time()
                valid_items = []
                for it in self._queue:
                    if it.deadline_ts > now:
                        valid_items.append(it)
                self._queue = valid_items

                if self._queue:
                    item = self._queue.pop(0)
                    if not self._queue:
                        self._has_items.clear()
                    return item
                else:
                    self._has_items.clear()

            await asyncio.sleep(0.05)
        return None

    def size(self) -> int:
        return len(self._queue)

