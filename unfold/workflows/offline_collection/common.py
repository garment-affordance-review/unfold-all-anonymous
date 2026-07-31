"""Shared helpers for offline collection pipeline."""

from __future__ import annotations

import logging
import signal
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def configure_runtime_warnings() -> None:
    warnings.filterwarnings("ignore", message=".*particle cloth.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Changing particle cloth mesh.*")
    logging.getLogger("isaacsim.core.prims.impl.cloth_prim").setLevel(logging.ERROR)
    logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)


def install_exit_signal_handlers(message: str) -> None:
    def _signal_handler(sig, frame):
        print(f"\n[INFO] {message}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
