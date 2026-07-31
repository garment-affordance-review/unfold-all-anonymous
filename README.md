# Anonymous Code Release

This repository contains the anonymous implementation for a two-stage garment
flattening pipeline. The method first learns a structural pair-value teacher
from large-scale physics-only simulation, then distills the teacher into an
RGB visual affordance model that predicts an ordered pair of grasp points.

Project page source files are under `docs/`. Enable GitHub Pages from the
`main` branch and `/docs` folder to publish the anonymous project page.

The release excludes paper sources, local datasets, logs, checkpoints, robot
calibration files, and real-robot run artifacts. Full-scale reproduction
requires external garment assets, generated supervision data, and pretrained
model checkpoints.

## Pipeline Overview

The main workflow has five stages:

1. Collect offline pair-reward data in Isaac Sim / IsaacLab.
2. Preprocess the offline data and train a Pointcept-based pair-value teacher.
3. Collect rendered RGB/depth/mask/geometry data from the same simulated assets.
4. Build image-space supervision from the teacher and train the visual model.
5. Run inference to predict ordered grasp-point heatmaps and decode a grasp pair.

Data flows through the system as:

```text
garment assets
  -> pair-conditioned Isaac rollouts
  -> point-cloud pair-reward dataset
  -> Pointcept pair-value teacher
  -> rendered RGB + geometry dataset
  -> teacher-generated A1/A2 heatmap supervision
  -> visual affordance model
  -> ordered grasp-pair inference
```

## Repository Layout

- `apps/`: thin command-line entry points for each major pipeline stage.
- `configs/`: Isaac simulation, offline collection, and rendering configs.
- `unfold/`: core environment, simulation, reward, storage, rendering, and
  learning modules.
- `tools/`: asset processing, supervision building, visualization, and
  evaluation utilities.
- `experiments/`: runnable experiment configs and helper scripts.

## Requirements

The simulation and rendering stages require Isaac Sim / IsaacLab. Training
uses PyTorch and standard scientific Python packages. The Pointcept teacher
stage expects an external Pointcept installation; configure its code, data,
and experiment paths locally.

Common dependencies include:

- Isaac Sim / IsaacLab
- PyTorch
- NumPy, SciPy, h5py, Pillow, PyYAML
- Gymnasium
- segmentation-model and point-cloud backbones used by the selected configs

All paths in configs are examples or placeholders. Replace dataset roots,
Pointcept roots, output directories, and checkpoint paths with local paths
before running.

## Stage 1: Offline Pair-Reward Collection

This stage evaluates ordered grasp pairs in physics-only simulation. For each
garment, the collector samples candidate vertex pairs, initializes the garment
conditioned on the selected pair, executes the directional loading sequence,
and stores the resulting reward.

Main entry points:

```bash
python apps/collect_pair_conditioned_offline_dataset.py \
  --config configs/offline_pair_conditioned.yaml \
  --num-envs 8
```

For multi-GPU or multi-worker collection:

```bash
python apps/collect_distributed.py \
  --config configs/offline_pair_conditioned_distributed.yaml
```

Relevant implementation:

- `unfold/workflows/offline_collection/pair_conditioned_collect.py`
- `unfold/workflows/offline_collection/distributed_collect.py`
- `unfold/platform/rewards.py`
- `unfold/simulation/control/unfold.py`

Expected output is a point-cloud/pair-reward dataset organized by garment
asset. Large generated datasets are intentionally not included in this release.

## Stage 2: Pointcept Teacher Preprocessing and Training

The offline collection output is converted into the format consumed by a
Pointcept-style point-cloud teacher. The teacher predicts the reward of an
ordered vertex pair from normalized garment geometry.

Typical preprocessing utilities:

```bash
python apps/rebuild_cloth_asset_index.py --data-root <GARMENT_DATA_ROOT>
python apps/build_pair_policy_offline_cache.py --config <CACHE_CONFIG>
```

The teacher training itself is handled in an external Pointcept workspace. Use
the offline dataset generated in Stage 1 as the Pointcept data root, and train
a pair-value model with a PT-v3m2-style backbone plus a pair readout head.

Relevant repository-side code:

- `unfold/algorithms/supervision/teacher_pointcept.py`
- `tools/pair_transfer/evaluate_seen_asset_unseen_pair.py`
- `tools/pair_transfer/evaluate_extra_unseen_pairs_online.py`
- `apps/eval_teacher_policy.py`

## Stage 3: Rendered Data Collection

This stage renders RGB observations and geometry annotations from simulated
garments. It reuses the Isaac environment, then records RGB/depth/masks,
camera parameters, visible vertices, and pixel-to-vertex correspondence needed
for later teacher distillation.

Example command:

```bash
python apps/render_dataset.py \
  --config configs/render_replicator.yaml \
  --headless
```

Relevant implementation:

- `unfold/workflows/rendering/app.py`
- `unfold/workflows/rendering/pipeline.py`
- `unfold/workflows/rendering/epoch.py`
- `unfold/workflows/rendering/io.py`
- `unfold/workflows/rendering/randomization.py`

The rendered dataset is large and should be stored outside the Git repository.

## Stage 4: Visual Supervision and Model Training

The trained Pointcept teacher is queried on visible garment vertices from the
rendered dataset. Teacher values are projected into image space and converted
into ordered heatmap supervision:

- `A1(x)`: first-grasp affordance.
- `A2(y | x1)`: conditional second-grasp affordance given the first query.

Build render supervision:

```bash
python apps/build_render_supervision.py \
  --config experiments/pair_policy/configs/render/supervision.yaml
```

Train the visual affordance model:

```bash
python apps/train_pair_policy.py \
  --config <VISUAL_TRAIN_CONFIG>
```

Relevant implementation:

- `unfold/workflows/render_supervision.py`
- `unfold/workflows/pair_policy_train.py`
- `unfold/algorithms/pair_policy/`
- `tools/pair_policy/build_supervision_minimal.py`
- `tools/pair_policy/visualize_supervision.py`

Training configs under `experiments/pair_policy/configs/train/` define the
backbone, target normalization, loss, augmentation, and output location.

## Stage 5: Inference

At inference time, the visual model predicts `A1` and conditional `A2`
heatmaps from a single RGB observation. A grasp pair is decoded by selecting a
first query from `A1`, evaluating the conditional second heatmap, and choosing
the ordered pair used by the downstream manipulation primitive.

Useful utilities:

```bash
python tools/pair_policy/infer_real_images_with_sam3.py --help
python tools/pair_policy/infer_video_with_sam3_pair_policy.py --help
```

For image-dataset evaluation and qualitative inspection:

```bash
python tools/pair_policy/eval_keypoints_alignment.py --help
python tools/pair_policy/build_artf_qualitative_grid.py --help
```

Real-robot calibration and execution scripts are excluded from this anonymous
release. The repository keeps only the simulation, training, rendering, and
offline evaluation code needed to inspect the method.

## Practical Notes

- Keep generated datasets, checkpoints, logs, and rendered frames outside Git.
- Replace placeholder paths in configs before running.
- Start with small `num_envs`, asset subsets, and short runs to validate the
  environment before launching large Isaac jobs.
- The full pipeline is computationally heavy; stages can be validated
  independently with small assets and sample counts.
