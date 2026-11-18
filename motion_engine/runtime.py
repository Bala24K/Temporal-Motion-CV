"""Latest-wins handoff for frames, ordered and lossless for control messages.

A queue would absorb bursts and then feed the analyser stale frames, so frames use a single slot: a newer frame
replaces an unread one and the drop is counted, never hidden.

Control messages (select exercise, end set, ...) must never be dropped. They used to share the frame slot, so a
frame arriving before the consumer ran silently discarded an exercise switch (docs/PROJECT_AUDIT.md, P1). They now
go to a FIFO that is always drained before the frame slot.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque, Optional


class LatestSlot:
    def __init__(self):
        self._item: Optional[Any] = None
        self._ctrl: Deque[Any] = deque()
        self._event = asyncio.Event()
        self.dropped = 0
        self.closed = False

    def put(self, item: Any) -> None:
        """Offer a frame; replaces (and counts) any unread frame."""
        if self._item is not None:
            self.dropped += 1
        self._item = item
        self._event.set()

    def put_control(self, item: Any) -> None:
        """Enqueue a control message; delivered in order, before any pending frame, never dropped."""
        self._ctrl.append(item)
        self._event.set()

    def close(self) -> None:
        self.closed = True
        self._event.set()

    async def get(self) -> Optional[Any]:
        """Next control message if any, else the latest frame. None once closed and drained."""
        while True:
            await self._event.wait()
            if self._ctrl:
                item = self._ctrl.popleft()
            else:
                item, self._item = self._item, None
            if not self._ctrl and self._item is None and not self.closed:
                self._event.clear()
            if item is not None or self.closed:
                return item
