#!/usr/bin/env python3
"""Deprecated entrypoint kept only to fail fast with the new unified pipeline."""


def main() -> None:
    raise RuntimeError(
        "build_pair_policy_offline_cache has been removed. "
        "Use apps/build_render_supervision.py to generate direct shard-backed training data."
    )


if __name__ == "__main__":
    main()
