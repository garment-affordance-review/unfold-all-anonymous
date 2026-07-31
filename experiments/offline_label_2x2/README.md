# Offline Label 2x2 Experiments

This area stores the offline-label ablation that separates the two design factors in the collection protocol:

- initialization:
  - random initialization
  - pair-conditioned initialization
- loading:
  - fling
  - directional `Y`-gravity

The goal is to test whether the final offline-label pipeline is reasonable in a simple and causal way.

## Main Question

Does the proposed offline labeling protocol produce labels that are:

- more repeatable under repeated execution of the same ordered pair
- less likely to underestimate high-potential grasp pairs

## Protocol Matrix

This experiment uses a full `2 x 2` design.

| ID | Initialization | Loading | Short name |
| --- | --- | --- | --- |
| A | random | fling | `random_fling` |
| B | random | `Y`-gravity | `random_y` |
| C | conditioned | fling | `cond_fling` |
| D | conditioned | `Y`-gravity | `cond_y` |

Interpretation:

- `A -> B` isolates the effect of replacing fixed fling with directional loading while keeping random initialization
- `B -> D` isolates the effect of replacing random initialization with pair-conditioned initialization while keeping directional loading
- `A -> C` isolates the effect of initialization under fling loading
- `C -> D` isolates the effect of loading under conditioned initialization
- `A -> D` compares the baseline to the final full method

## Dataset And Evaluation Unit

Evaluation should be performed on:

- the same asset set
- the same ordered pair set per asset
- repeated executions for each `(asset, ordered_pair, protocol)` tuple

The atomic evaluation unit is:

- one ordered pair on one asset under one protocol repeated `N` times

## Fixed Evaluation Budget

Use the following fixed setting for the main experiment.

- assets: `100`
- asset sampling rule: stratified by point-count distribution only
- no extra category balancing is required once the point-count distribution is controlled
- anchor construction: farthest point sampling to `128` anchors per asset
- pair construction: ordered anchor pairs
- pair stratification: `4` pair-distance bins
- pairs per bin: `8`
- total evaluation pairs per asset: `32`
- repeats per `(asset, pair, protocol)`: `8`
- protocols: `4`

Total number of executions:

- `4 protocols x 100 assets x 32 pairs x 8 repeats = 102,400`

Important terminology:

- the `128` number refers to FPS anchor points
- the actual executed evaluation set is `32` ordered pairs per asset

## What This Experiment Should Prove

This experiment is designed to support two claims.

### Claim 1: Random initialization introduces label noise

Expected evidence:

- `cond_fling` is more repeatable than `random_fling`
- `cond_y` is more repeatable than `random_y`

The intended interpretation is:

- uncontrolled wrinkles, entanglement, and drape variation corrupt offline pair labels
- pair-conditioned initialization reduces this source of noise

### Claim 2: Fixed fling parameters underestimate some high-potential pairs

Expected evidence:

- `random_y` achieves higher best repeated reward than `random_fling`
- `cond_y` achieves higher best repeated reward than `cond_fling`

The intended interpretation is:

- the fixed fling rollout can fail to reveal the attainable quality of some pairs
- directional loading is a better probe of structural unfold potential

## Primary Metrics

Use one repeatability metric and one potential metric as the main results.

### 1. Reward Std

For each `(asset, pair, protocol)`:

- run repeated trials
- compute reward standard deviation across repeats

This is the main noise metric.

Desired direction:

- lower is better

### 2. Best-of-N Reward

For each `(asset, pair, protocol)`:

- run repeated trials
- take the maximum reward across repeats

This is the main potential metric.

Desired direction:

- higher is better

## Secondary Metrics

Use only if needed to support the main story.

- mean reward
- reward range
- pairwise rank consistency across repeats
- top-k overlap across repeats

## Recommended Main Result Table

This should be the primary table in the experiment note and later in the paper draft.

| Protocol | Reward Std ↓ | Best-of-N Reward ↑ | Mean Reward ↑ | Rank Consistency ↑ |
| --- | ---: | ---: | ---: | ---: |
| `random_fling` | `...` | `...` | `...` | `...` |
| `random_y` | `...` | `...` | `...` | `...` |
| `cond_fling` | `...` | `...` | `...` | `...` |
| `cond_y` | `...` | `...` | `...` | `...` |

Target reading pattern:

- noise effect:
  - compare `random_fling` vs `cond_fling`
  - compare `random_y` vs `cond_y`
- loading effect:
  - compare `random_fling` vs `random_y`
  - compare `cond_fling` vs `cond_y`
- final method:
  - compare `random_fling` vs `cond_y`

## Recommended Figures

Keep the figure set minimal.

### Figure 1: Reward Std Boxplot

Plot:

- x-axis: the four protocols
- y-axis: per-pair reward std

Purpose:

- directly show the noise reduction from conditioned initialization

### Figure 2: Best-of-N Reward Boxplot

Plot:

- x-axis: the four protocols
- y-axis: per-pair best-of-N reward

Purpose:

- directly show whether `Y`-gravity reveals higher attainable pair quality than fixed fling

## Minimal Output Layout

This experiment group should keep:

- `scripts/`
  - runners and analysis scripts for the 2x2 protocol
- `runs/`
  - raw outputs, summaries, plots, and per-run notes

Suggested future run layout:

- `runs/<run_name>/raw/`
- `runs/<run_name>/summary/metrics.csv`
- `runs/<run_name>/summary/main_table.md`
- `runs/<run_name>/plots/reward_std_boxplot.png`
- `runs/<run_name>/plots/best_of_n_boxplot.png`

## Unified Entrypoint

Run a pilot with one command:

```bash
${CONDA_ENVS_ROOT}/isaac/bin/python \
experiments/offline_label_2x2/scripts/run_experiment.py \
  --device cuda:0 \
  --mode generate \
  --run-name pilot_fixed_inputs \
  --num-assets 8 \
  --asset-bins 4 \
  --anchor-count 32 \
  --pair-distance-bins 4 \
  --pairs-per-bin 2 \
  --protocol all \
  --num-envs 8 \
  --repeats-per-pair 4 \
  --overwrite
```

This unified entrypoint:

- generates a fixed asset manifest
- generates a fixed evaluation-pair manifest
- writes `run_config.json`
- launches the repeatability runner on those fixed inputs
- uses the experiment-local config:
  - [`configs/offline_label_2x2.yaml`](${PROJECT_ROOT}/experiments/offline_label_2x2/configs/offline_label_2x2.yaml)

Use `--mode reuse` with `--assets-manifest` and `--pairs-manifest` to rerun exactly the same fixed inputs.

## Recording One Visual Check

Use the dedicated recorder to save two mp4 views per protocol on the same `(asset, pair)`:

```bash
${CONDA_ENVS_ROOT}/isaac/bin/python \
experiments/offline_label_2x2/scripts/record_protocol_videos.py \
--device cuda:0 \
--enable_cameras \
--headless \
--protocol all \
--asset-indices 0 \
--num-envs 1 \
--num-pairs 4 \
--pair-index 0 \
--output-dir experiments/offline_label_2x2/runs/recordings \
--overwrite
```

Notes:

- recording now uses two fixed headless render-product cameras and does not require a visible viewport
- this script records only one environment and one pair for clean visual inspection
- outputs are written under `runs/recordings/asset_xxxx/pair_xx/<protocol>/{side,top}/`
- each protocol and each view also saves raw PNG frames next to the mp4 for debugging encoder issues

## Suggested Sampling Note

To keep the comparison fair:

- first sample `100` assets by point-count distribution
- then, for each selected asset, build the same `128`-anchor candidate space
- then sample the same `32` evaluation pairs from the same `4` distance bins
- finally execute all four protocols on this shared evaluation set

## Current Status

Current contents:

- design note: this file
- experiment config:
  - [offline_label_2x2.yaml](${PROJECT_ROOT}/experiments/offline_label_2x2/configs/offline_label_2x2.yaml)
- implementation note:
  - [IMPLEMENTATION_PLAN.md](${PROJECT_ROOT}/experiments/offline_label_2x2/IMPLEMENTATION_PLAN.md)
- protocol runner:
  - [run_protocol_repeatability.py](${PROJECT_ROOT}/experiments/offline_label_2x2/scripts/run_protocol_repeatability.py)

The runner currently supports:

- fixed-asset repeated evaluation
- `all` or single-protocol execution
- protocol switching across:
  - `random_fling`
  - `random_y`
  - `cond_fling`
  - `cond_y`
- per-run CSV / JSON output under `runs/`

Minimal validation completed:

- Python syntax compilation for the runner and updated `Unfold` controller

Not yet validated:

- end-to-end rollout inside the IsaacLab runtime
- numerical sanity of the new `fling` trajectory versus the old baseline intent

Planned next additions:

- summary script
- first pilot run results
