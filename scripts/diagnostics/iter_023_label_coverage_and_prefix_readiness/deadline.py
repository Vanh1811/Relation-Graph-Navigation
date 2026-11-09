"""Shared fail-closed wall-clock deadline for future authorized pilots."""
from __future__ import annotations

import time
from typing import Callable


class DeadlineExceeded(RuntimeError):
    pass


class DeadlineGuard:
    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if seconds <= 0:
            raise ValueError("deadline seconds must be positive")
        self.seconds = float(seconds)
        self.clock = clock
        self.started_at = clock()
        self.deadline_at = self.started_at + self.seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_at - self.clock())

    def assert_open(self) -> None:
        if self.remaining_seconds() <= 0:
            raise DeadlineExceeded(f"pilot wall-clock limit of {self.seconds:g} seconds reached")
