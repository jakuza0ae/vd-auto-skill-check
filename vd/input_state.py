"""Input state and timestamps shared by the platform backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MouseState:
    held: bool
    healthy: bool
    device: str | None
    error: str | None


@dataclass(frozen=True)
class Keydown:
    requested_at: float
    syn_completed_at: float


class InputDisarmed(RuntimeError):
    """The output guard changed after the dispatch gate claimed an attempt."""


class InputDeadlineMissed(RuntimeError):
    """The final input guard finished after a claimed attempt expired."""
