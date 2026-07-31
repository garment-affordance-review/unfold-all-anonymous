# Offline Teacher Evaluation Experiments

This directory owns evaluation-only experiments for the Pointcept offline teacher.

It is separate from `experiments/pair_policy`, which owns supervision generation and RGB student training.

Layout:

- `scripts/run_extra_unseen_pairs_online.sh`: online evaluation launcher for extra unseen pairs on seen assets
- `runs/`: evaluation outputs and logs

Current active evaluation:

1. launch `scripts/run_extra_unseen_pairs_online.sh`
2. inspect the generated summaries under `runs/`
