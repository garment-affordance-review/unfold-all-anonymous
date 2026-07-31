# Paper Context For This Experiment

This note ties the external zero-shot benchmark to the current project narrative.

## Current Project Framing

The current pair-policy pipeline predicts:

- `A1(x)`: first-grasp affordance
- `A2(y | x1)`: conditional second-grasp affordance

See:

- [`docs/PAIR_POLICY_EXPERIMENT_WORKFLOW.md`](${PROJECT_ROOT}/docs/PAIR_POLICY_EXPERIMENT_WORKFLOW.md)
- [`outputs/paper_figures/paper_figure_draft`](${PROJECT_ROOT}/outputs/paper_figures/paper_figure_draft)

The intended claim is not only that the model can localize graspable cloth points, but that it can model ordered grasp-pair quality.

## Why The Cloth Competition Benchmark Is Useful

This benchmark is a strong external test because it is:

- outside our own render/data-generation pipeline
- directly about cloth unfolding grasp selection
- naturally aligned with pair-selection evaluation

That makes it a better zero-shot test than another in-domain synthetic split.

## The Specific Claim We Want To Test

The most important claim for this experiment is:

- conditional pair modeling matters

In benchmark terms, this becomes:

- does `A2(y | x1)` help beyond selecting two individually strong points?

This should be reflected in all result tables and ablations.

## Recommended Paper-Friendly Ablations

At minimum:

- full model: `A1 + conditional A2`
- ablation: `A1 + independent A2`
- baseline: random valid pair

If benchmark labels allow it:

- benchmark/reference pair
- simple geometry heuristic

## Recommended Result Story

The cleanest zero-shot story is:

1. our model transfers to an external cloth benchmark without finetuning
2. `A1` alone is not enough
3. conditional `A2` improves pair quality or pair ranking
4. even when absolute performance is imperfect, the ranking signal remains useful and non-random

## What Would Count As A Positive Result

Any of the following would be meaningful:

- lower pair distance than random and non-conditional baselines
- higher top-k recall of benchmark grasp pairs
- stronger ranking correlation with benchmark action quality
- better replay reward in our simulator than random or independent-point decoding

## What Would Count As A Weak But Still Useful Result

Even if absolute transfer is limited, the experiment is still valuable if it shows:

- `A1` remains somewhat stable zero-shot
- `A2(y | x1)` degrades less than naive independent decoding
- failure cases expose a concrete domain-gap explanation

That would still help sharpen the paper's limitations section.
