"""Policy exports with lazy imports to avoid pulling heavy dependencies on package import."""

from __future__ import annotations

from typing import Any

__all__ = ["OfflineReplayPolicy", "RandomPolicy"]


def __getattr__(name: str) -> Any:
    if name == "OfflineReplayPolicy":
        from .offline_replay_policy import OfflineReplayPolicy

        return OfflineReplayPolicy
    if name == "RandomPolicy":
        from .random_policy import RandomPolicy

        return RandomPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

