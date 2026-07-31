# Experiment Plan

This document is the working execution plan for zero-shot evaluation on the cloth competition benchmark.

## Scope Choice

Current scope choice:

- `minimal fix` equivalent on the experiment side

Meaning:

- do not redesign the policy first
- do not finetune on benchmark data first
- start with pure zero-shot inference and benchmark-grounded evaluation

## Phase 0: Benchmark Intake

Inputs:

- extracted benchmark files under `${DATASET_ROOT}/cloth_competition/`

Deliverables:

- folder tree snapshot
- annotation schema note
- sample manifest

Questions to answer:

- which image file should be fed to the pair-policy model
- whether a usable mask already exists
- where grasp annotations live
- whether grasps are ordered

## Phase 1: Inference Adapter

Goal:

- define a deterministic path from benchmark sample to `A1/A2` prediction

Needed decisions:

- resize/crop convention
- mask source:
  - benchmark mask
  - external segmenter
  - fallback heuristic
- decode rule from heatmaps to an ordered pair

Expected outputs:

- per-sample prediction files
- visual overlays for sanity check

## Phase 2: Annotation Alignment Metrics

Main metrics:

- pair distance to benchmark annotation
- top-k pair recall
- first-point hit rate
- conditional second-point hit rate

Baselines:

- random valid pair
- independent-point decode

## Phase 3: Ranking Metrics

Run this only if the benchmark annotations support it.

Main metrics:

- AUROC
- AP
- rank correlation
- NDCG

## Phase 4: Optional Simulator Replay

Run this only if benchmark state/action semantics are close enough.

Main metrics:

- unfold reward
- reward gain over random
- best-of-k candidate success

## Minimal Pilot

Before full evaluation, run a trusted pilot subset with:

- `16` to `64` samples
- saved visual overlays
- raw decoded pairs
- manual inspection of obvious failures

The pilot should answer:

- are coordinate systems correct
- are masks aligned
- is pair ordering interpreted correctly
- is the model obviously off-distribution

## Output Convention

Suggested run names:

- `pilot_manifest_check`
- `pilot_zero_shot_decode_v1`
- `full_zero_shot_align_v1`
- `full_zero_shot_rank_v1`

Suggested output root:

- `experiments/cloth_competition_zero_shot/runs/<run_name>/`

Suggested summary root:

- `experiments/cloth_competition_zero_shot/analysis/<run_name>/`
