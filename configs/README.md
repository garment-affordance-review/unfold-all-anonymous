# Config Layout

Root `configs/` now keeps only repository-wide runtime configs that are used by
general entrypoints:

- `config.yaml`: generic teacher-eval environment config
- `offline_standard.yaml`: generic random offline collection config
- `offline_pair_conditioned.yaml`: generic pair-conditioned offline collection config
- `offline_pair_conditioned_distributed.yaml`: distributed pair-conditioned collection config
- `render_replicator.yaml`: render dataset generation config

Experiment-owned configs should live next to the experiment that uses them.

Current examples:

- `experiments/offline_teacher_eval/scripts/run_extra_unseen_pairs_online.sh`
- `experiments/offline_label_2x2/configs/offline_label_2x2.yaml`
- `experiments/pair_policy/configs/render_pair_policy_supervision.yaml`
- `experiments/pair_policy/configs/train_render_pair_policy_full.yaml`

Older pair-policy variants that used to live in root `configs/` were moved to:

- `experiments/pair_policy/configs/legacy/`

Those legacy files are kept for reference and should not be used as new default
entrypoints.
