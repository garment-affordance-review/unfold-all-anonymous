# aRTFClothes Experiment Notes

This directory keeps the main aRTF-Clothes experiment outputs used by the paper:

- `analysis/pair_policy_keypoint_eval_full`
- `analysis/clothmate_keypoint_eval_full_hcenter`
- `analysis/clothmate_vs_pair_policy_full_hcenter`
- `analysis/affordance_behavior_metrics`
- `analysis/paper_summary`
- `figures/qualitative_grid_3x4_rows_v3*`
- `figures/qualitative_grid_3x9_rows_v3*`
- `runs/full_cropfit_dual`

## aRTF Zero-Shot Evaluation

The current zero-shot sim-to-real comparison uses:

- dataset: `aRTF-Clothes`
- splits: `train`, `test`, and `combined`
- categories: `towels`, `tshirts`, `shorts`
- methods:
  - `pair_policy_keypoint_eval_full`
  - `clothmate_keypoint_eval_full_hcenter`

The current comparison table is stored under:

- `analysis/clothmate_vs_pair_policy_full_hcenter`

## Working Semantic Rules

The following rules are a working definition for future semantic-to-flattening analysis.
They are not yet treated as a finalized benchmark protocol.

We define two nested success notions:

- `Rigid-positive`: the predicted pair matches a stable global support structure.
- `Deformable-positive`: the predicted pair can still induce a meaningful local or global opening.

By design:

- `Rigid-positive` is always also `Deformable-positive`.
- `Deformable-positive` is a superset of `Rigid-positive`.

### Towel

Raw semantic points:

- `corner0`, `corner1`, `corner2`, `corner3`

Rules:

- `Rigid-positive`
  - any adjacent-corner pair
- `Deformable-positive`
  - any adjacent-corner pair
- `Negative`
  - diagonal-corner pairs
  - same-corner pairs

Interpretation:

- Adjacent-corner grasps align with the desired edge-based flattening behavior.
- Diagonal grasps are not counted as success for either metric.

### T-shirt

Raw semantic points:

- `shoulder_left`, `neck_left`, `neck_right`, `shoulder_right`
- `sleeve_right_top`, `sleeve_right_bottom`
- `armpit_right`, `waist_right`
- `waist_left`, `armpit_left`
- `sleeve_left_bottom`, `sleeve_left_top`

Semantic regions:

- `top band`
  - `shoulder_left`
  - `neck_left`
  - `neck_right`
  - `shoulder_right`
  - `sleeve_left_top`
  - `sleeve_right_top`
- `side-lower`
  - `armpit_left`
  - `armpit_right`
  - `sleeve_left_bottom`
  - `sleeve_right_bottom`
- `bottom band`
  - `waist_left`
  - `waist_right`

Rules:

- `Rigid-positive`
  - both grasp points lie in the `top band`
  - the pair spans left/right structure rather than remaining strictly local
  - representative examples:
    - `shoulder_left + shoulder_right`
    - `shoulder_left + neck_right`
    - `neck_left + shoulder_right`
    - `neck_left + neck_right`
    - `sleeve_left_top + sleeve_right_top`
    - `shoulder_left + sleeve_right_top`
    - `sleeve_left_top + shoulder_right`

- `Deformable-positive`
  - all `Rigid-positive` pairs
  - same-side top-band pairs that still support local opening
  - representative examples:
    - `shoulder_left + neck_left`
    - `shoulder_left + sleeve_left_top`
    - `neck_left + sleeve_left_top`
    - right-side symmetric cases
  - limited top-band to side-lower pairs that still induce local opening
  - representative examples:
    - `shoulder_left + armpit_left`
    - `sleeve_left_top + armpit_left`
    - `shoulder_right + armpit_right`
    - `sleeve_right_top + armpit_right`

- `Negative`
  - pairs concentrated on the `bottom band`
  - pairs dominated by lower sleeve or lower side structure without top-band support
  - representative examples:
    - `waist_left + waist_right`
    - `waist + armpit`
    - `sleeve_bottom + waist`

Interpretation:

- `Rigid-positive` measures whether the grasp pair covers the upper support structure in a globally stabilizing way.
- `Deformable-positive` also allows more local opening behavior on the same side.

### Shorts

Raw semantic points:

- `waist_left`, `waist_right`
- `pipe_right_outer`, `pipe_right_inner`
- `crotch`
- `pipe_left_inner`, `pipe_left_outer`

Semantic regions:

- `top band`
  - `waist_left`
  - `waist_right`
- `outer support`
  - `pipe_left_outer`
  - `pipe_right_outer`
- `inner structure`
  - `pipe_left_inner`
  - `pipe_right_inner`
  - `crotch`

Rules:

- `Rigid-positive`
  - `waist_left + waist_right`

- `Deformable-positive`
  - all `Rigid-positive` pairs
  - same-side top-to-outer opening pairs
    - `waist_left + pipe_left_outer`
    - `waist_right + pipe_right_outer`
  - cross-outer support pairs
    - `pipe_left_outer + pipe_right_outer`

- `Negative`
  - pairs dominated by `inner structure`
  - representative examples:
    - `pipe_inner + pipe_inner`
    - `crotch + pipe_inner`
    - `crotch + crotch`
  - `waist + pipe_inner` is currently treated as negative in this working definition

Interpretation:

- `waist_left + waist_right` is treated as the cleanest rigid support pair.
- `waist + outer leg` and `outer leg + outer leg` are still allowed as deformable-effective opening behaviors.

## Status

These rules are recorded as a working experimental definition only.
They should be refined against qualitative inspection and, if needed, later aligned with a more formal semantic-to-flattening metric.
