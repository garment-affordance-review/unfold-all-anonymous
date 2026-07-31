# Offline Label 2x2 Minimal Implementation Plan

This note records the minimal code changes required to run the `2 x 2` offline-label experiment cleanly in the current repository.

It is implementation-focused. The experiment design itself is documented in [`README.md`](${PROJECT_ROOT}/experiments/offline_label_2x2/README.md).

## Goal

Support the following four protocols on the same `(asset, ordered_pair)` evaluation set:

- `random_fling`
- `random_y`
- `cond_fling`
- `cond_y`

The implementation should:

- reuse existing project code as much as possible
- avoid duplicating per-protocol scripts
- keep the protocol definition explicit and symmetric

## What Already Exists

### A. Conditioned initialization

Already implemented in:

- [`unfold/workflows/offline_collection/pair_conditioned_collect.py`](${PROJECT_ROOT}/unfold/workflows/offline_collection/pair_conditioned_collect.py)

Relevant function:

- `_apply_pair_conditioned_poses(...)`

This already provides:

- per-pair deterministic pose initialization
- optional small rotation noise
- clean direct control over the cloth pose before action execution

### B. Random initialization

Already implemented in the environment reset path:

- [`unfold/simulation/env.py::_reset_idx`](${PROJECT_ROOT}/unfold/simulation/env.py)

Current behavior:

1. sample random pose
2. `reset_to_poses(...)`
3. stabilize
4. optionally call `_apply_predrop_relift(...)`
5. stabilize again

Important finding:

- the code supports two-stage randomization through `_apply_predrop_relift(...)`
- but current offline configs do not enable `spawn_cfg.predrop_relift.enabled`

So:

- the capability exists
- the experiment runner must enable it explicitly for `random_*` protocols

### C. Directional `Y`-gravity loading

Already implemented in:

- [`unfold/simulation/control/unfold.py`](${PROJECT_ROOT}/unfold/simulation/control/unfold.py)

Current `Unfold.step()` already uses:

- lift
- lateral `+y` gravity during stretch/hold
- later restore normal `-z` gravity

This is the current `cond_y` protocol.

### D. Fixed-pair repeat execution scaffold

Already implemented in:

- [`unfold/workflows/offline_collection/pair_repeatability.py`](${PROJECT_ROOT}/unfold/workflows/offline_collection/pair_repeatability.py)

This file is the best starting point for the 2x2 runner because it already supports:

- fixed asset
- fixed repeated pairs
- repeated reward collection
- CSV / JSON / plot outputs

## What Is Missing

### 1. Explicit protocol abstraction

The repository currently has:

- two initialization paths
- one mature loading path

But it does not yet expose:

- `init_mode = random | conditioned`
- `loading_mode = fling | y_gravity`

as explicit protocol switches.

### 2. `fling` as a loading-mode option

The current `Unfold` controller hardcodes the `Y`-gravity loading sequence.

To support `random_fling` and `cond_fling`, the loading phase must be switchable.

### 3. A unified experiment runner

We need one runner that:

- fixes the evaluation asset set
- fixes the evaluation pair set
- loops over protocols
- repeats each `(asset, pair, protocol)` `8` times
- writes results in one experiment-specific format

## Minimal Design

The minimal design is to isolate exactly two axes:

- `init_mode`
- `loading_mode`

and drive both from a single experiment runner.

### Protocol definition

| Protocol | init_mode | loading_mode |
| --- | --- | --- |
| `random_fling` | `random` | `fling` |
| `random_y` | `random` | `y_gravity` |
| `cond_fling` | `conditioned` | `fling` |
| `cond_y` | `conditioned` | `y_gravity` |

## Minimal Code Changes

### Change 1: Add explicit loading mode to `Unfold`

Primary file:

- [`unfold/simulation/control/unfold.py`](${PROJECT_ROOT}/unfold/simulation/control/unfold.py)

Recommended change:

- add a small config field such as `action_sequence.loading_mode`
- supported values:
  - `y_gravity`
  - `fling`

Current default should remain:

- `y_gravity`

This keeps existing behavior unchanged.

### Change 2: Route the loading phases by mode

Keep the overall sequence structure the same:

- grasp
- lift
- stretch / load
- hold / release
- stabilize

But make the loading-specific part switchable.

Recommended split:

- `_execute_lifting_phase(...)`
- `_execute_loading_phase(...)`
- `_execute_holding_phase(...)`

Then:

- `y_gravity` mode:
  - preserve current gravity-based loading behavior
- `fling` mode:
  - use the same vertex-control system
  - replace the current loading body with a short waypoint-based fling trajectory

This avoids forking the whole controller.

### Change 3: Add explicit init helpers in the experiment runner

Do not push the 2x2 experiment logic into `Env._reset_idx`.

Instead, in the experiment runner, add two explicit helpers:

- `_apply_random_init(...)`
- `_apply_conditioned_init(...)`

Recommended behavior:

#### `_apply_random_init(...)`

- call environment random reset logic
- explicitly enable two-stage randomization:
  - random pose drop
  - predrop relift
  - re-stabilize

Important:

- only random protocols should use relift

#### `_apply_conditioned_init(...)`

- reuse `_apply_pair_conditioned_poses(...)`
- do not apply relift afterward

This preserves the meaning of conditioned initialization.

### Change 4: Add a dedicated experiment runner under `experiments/offline_label_2x2/scripts`

Recommended new script:

- `experiments/offline_label_2x2/scripts/run_protocol_repeatability.py`

This runner should:

1. load the fixed asset list
2. build the fixed pair list for each asset
3. iterate over all four protocols
4. for each `(asset, pair, protocol)`:
   - initialize with `init_mode`
   - execute with `loading_mode`
   - record reward
5. aggregate summary files

This keeps experiment-specific logic inside the experiment group rather than polluting the general offline collection workflow.

## Why Only Random Init Should Use Relift

This follows directly from the protocol semantics.

### Random init

The goal is to create uncontrolled cloth-state variation.

So the protocol should use:

- random pose drop
- relift / jitter
- second free fall

This is the correct source of label noise for the `random_*` protocols.

### Conditioned init

The goal is to place the cloth in a pair-conditioned pose.

If relift is applied afterward:

- the protocol is no longer pair-conditioned in a clean sense
- the initialization semantics become mixed

Therefore:

- `conditioned` protocols should never use relift

## `fling` Minimal Implementation Strategy

The project already has the right low-level primitive:

- tensor-based vertex trajectory control in [`unfold/simulation/control/action.py`](${PROJECT_ROOT}/unfold/simulation/control/action.py)

So `fling` does not need a new control system.

It only needs:

- a small waypoint-based vertex motion path

The implementation should stay minimal:

1. lift the two constrained vertices to the pre-fling height
2. stretch to the configured separation
3. execute a short fling trajectory using a few target waypoints
4. release
5. stabilize

The important design rule is:

- reuse the same action manager and phase loop
- only swap the loading body

## Recommended File Ownership

### Core controller

- [`unfold/simulation/control/unfold.py`](${PROJECT_ROOT}/unfold/simulation/control/unfold.py)
  - add `loading_mode`
  - add fling loading branch

### Experiment runner

- `experiments/offline_label_2x2/scripts/run_protocol_repeatability.py`
  - experiment-specific protocol loop
  - random vs conditioned initialization dispatch
  - result writing

### Optional helper extraction

If a helper becomes reusable, move it later. Not before it is justified.

For example:

- a generic `_apply_random_init_with_relift(...)`
- a generic fixed-pair repeat evaluator

But do not over-abstract on the first implementation.

## Minimal Acceptance Criteria

The implementation is sufficient when:

1. all four protocols can be run from one experiment runner
2. `random_*` protocols use two-stage randomization
3. `cond_*` protocols use direct pair-conditioned initialization without relift
4. `fling` and `y_gravity` are switchable without duplicating the full controller
5. the runner writes one flat result table with:
   - `asset_id`
   - `pair_id`
   - `protocol`
   - `repeat_idx`
   - `reward`

## Suggested Next Step

Implement in this order:

1. add `loading_mode` to `Unfold`
2. implement `random_y` and `cond_y` in the dedicated experiment runner
3. add the `fling` loading branch
4. complete the full `2 x 2` protocol sweep
