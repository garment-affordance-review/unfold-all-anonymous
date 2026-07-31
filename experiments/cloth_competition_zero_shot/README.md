# Cloth Competition Zero-Shot Experiments

This experiment area owns zero-shot evaluation of the current `A1/A2` visual affordance model on the external cloth-unfolding benchmark introduced by:

- `A Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection: The ICRA 2024 Cloth Competition`

This directory is intentionally separate from:

- [`experiments/pair_policy`](${PROJECT_ROOT}/experiments/pair_policy), which owns RGB student training
- [`experiments/offline_teacher_eval`](${PROJECT_ROOT}/experiments/offline_teacher_eval), which owns evaluation-only runs for the Pointcept offline teacher

The main purpose here is:

- evaluate whether the current `A1/A2` model transfers zero-shot to an external benchmark
- test whether conditional second-point prediction is useful beyond independent point selection
- prepare a clean place for benchmark-specific notes, manifests, scripts, and result summaries

## Quick Links

Dataset and local inspection:

- zip file: [ICRA_2024_cloth_competition_dataset.zip](${DATASET_ROOT}/cloth_competition/ICRA_2024_cloth_competition_dataset.zip)
- local source note: [ICRA_2024_cloth_competition_dataset_SOURCE.md](${DATASET_ROOT}/cloth_competition/ICRA_2024_cloth_competition_dataset_SOURCE.md)
- dataset notes: [docs/DATASET_NOTES.md](${PROJECT_ROOT}/experiments/cloth_competition_zero_shot/docs/DATASET_NOTES.md)
- debug views with annotated sample screenshots: [analysis/debug_views](${PROJECT_ROOT}/experiments/cloth_competition_zero_shot/analysis/debug_views)

Current experiment results:

- iteration summary: [analysis/iteration_20260416_single_grasp_proxy_summary.md](${PROJECT_ROOT}/experiments/cloth_competition_zero_shot/analysis/iteration_20260416_single_grasp_proxy_summary.md)
- full benchmark run: [runs/full_single_grasp_expkl_sam3_v1](${PROJECT_ROOT}/experiments/cloth_competition_zero_shot/runs/full_single_grasp_expkl_sam3_v1)
- pilot run with SAM3: [runs/pilot_single_grasp_expkl_sam3_v1](${PROJECT_ROOT}/experiments/cloth_competition_zero_shot/runs/pilot_single_grasp_expkl_sam3_v1)

## Main Question

Can the current image-space ordered grasp policy:

- predict useful first-grasp heatmaps `A1`
- predict useful conditional second-grasp heatmaps `A2(y | x1)`
- generalize zero-shot to the ICRA 2024 Cloth Competition distribution without retraining on that dataset

## Why This Experiment Matters

This is the cleanest external-distribution check for the current pair-policy story.

If the model works zero-shot here, that supports the claim that it learned:

- transferable first-point affordance
- transferable conditional pair structure

rather than only overfitting to our own rendered Isaac distribution.

## Minimal Evaluation Layers

We should treat this as a staged experiment, not one monolithic benchmark.

### 1. Annotation Alignment

Use benchmark observations and compare predicted pair locations against benchmark grasp annotations.

Primary outputs:

- pair distance to benchmark grasp pair
- top-k pair recall
- first-point hit rate
- conditional second-point hit rate

### 2. Pair Ranking

If the dataset exposes multiple candidate actions, rankings, or success labels, evaluate whether our predicted pair score ranks stronger actions higher.

Primary outputs:

- AUROC / AP for success-like labels
- rank correlation
- NDCG or top-k success recall

### 3. Closed-Loop Replay

If benchmark states can be mapped into our environment or approximated well enough:

- run predicted grasp pairs in Isaac
- compare unfold reward against baselines and benchmark/reference pairs

Primary outputs:

- reward improvement
- success rate
- best-of-k candidate success

## Required Baselines

At minimum, compare against:

- random valid pair
- independent top-1 point selection without conditional `A2`
- benchmark/annotated pair

Important ablation:

- conditional `A2(y | x1)`
- non-conditional second-point selection

This ablation is central because it tests whether ordered pair modeling is actually helping.

## Expected Output Layout

- `docs/`
  - benchmark notes
  - paper/context notes
  - experiment plan
- `scripts/`
  - benchmark conversion
  - zero-shot inference
  - evaluation and plotting
- `manifests/`
  - benchmark split manifests
  - sample subsets
- `runs/`
  - raw predictions, logs, and metrics
- `analysis/`
  - summaries, tables, and figures

## Suggested Run Layout

- `runs/<run_name>/predictions/`
- `runs/<run_name>/metrics.json`
- `runs/<run_name>/metrics.csv`
- `runs/<run_name>/console.log`
- `analysis/<run_name>/summary.md`
- `analysis/<run_name>/summary.json`
- `analysis/<run_name>/figures/`

## Immediate Next Steps

1. inspect the downloaded benchmark layout under `${DATASET_ROOT}/cloth_competition/`
2. document the exact annotation schema and split definition
3. build a benchmark manifest that maps each sample to image, mask, and grasp annotation
4. define the zero-shot decode protocol for converting `A1/A2` into an ordered grasp pair
5. run a small pilot on a trusted subset before scaling to the full benchmark
