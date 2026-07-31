# Pair Policy Experiments

This directory now keeps a small set of active training baselines for pair-policy comparison.

All active configs share the same current training setup:

- fixed shard dataset: `${DATA_ROOT}/pair_policy_train_v2/`
- invalid-fill patch already applied to shard value maps
- `supervision_mask_mode: input_mask`
- ImageNet input normalization
- mask augmentation
- RGB augmentation
- light geometric augmentation
- learning rate: `1e-4`

The active experiment matrix only varies:

- backbone
- target normalization
- loss

## Active Train Configs

- `configs/train/segformer_mit_b4_minmax_weighted_huber.yaml`
- `configs/train/segformer_mit_b4_minmax_bce.yaml`
- `configs/train/segformer_mit_b4_exp_kl_tau01.yaml`
- `configs/train/unetpp_resnet50_minmax_weighted_huber.yaml`

## What Each Active Config Tests

- `segformer_mit_b4_minmax_weighted_huber.yaml`
  - main baseline
  - compares the strongest current backbone with a regression-style objective

- `segformer_mit_b4_minmax_bce.yaml`
  - min-max target with pixelwise classification-style loss
  - checks whether BCE sharpens heatmaps or collapses A1/A2

- `segformer_mit_b4_exp_kl_tau01.yaml`
  - distribution-learning baseline
  - checks whether `exp + KL` works again after invalid-value cleanup

- `unetpp_resnet50_minmax_weighted_huber.yaml`
  - backbone control
  - isolates the benefit of `SegFormer(MIT-B4)` against a strong CNN encoder-decoder

## Run Commands

`SegFormer(MIT-B4) + min-max + weighted_huber`

```bash
PYTHON=${CONDA_ENVS_ROOT}/pointcept/bin/python \
bash experiments/pair_policy/scripts/run_train_pair_policy.sh \
experiments/pair_policy/configs/train/segformer_mit_b4_minmax_weighted_huber.yaml
```

`SegFormer(MIT-B4) + min-max + BCE`

```bash
PYTHON=${CONDA_ENVS_ROOT}/pointcept/bin/python \
bash experiments/pair_policy/scripts/run_train_pair_policy.sh \
experiments/pair_policy/configs/train/segformer_mit_b4_minmax_bce.yaml
```

`SegFormer(MIT-B4) + exp + KL (tau=0.1)`

```bash
PYTHON=${CONDA_ENVS_ROOT}/pointcept/bin/python \
bash experiments/pair_policy/scripts/run_train_pair_policy.sh \
experiments/pair_policy/configs/train/segformer_mit_b4_exp_kl_tau01.yaml
```

`UnetPlusPlus(ResNet50) + min-max + weighted_huber`

```bash
PYTHON=${CONDA_ENVS_ROOT}/pointcept/bin/python \
bash experiments/pair_policy/scripts/run_train_pair_policy.sh \
experiments/pair_policy/configs/train/unetpp_resnet50_minmax_weighted_huber.yaml
```

## Train Outputs

Expected output roots:

- `experiments/pair_policy/runs/train/segformer_mit_b4_minmax_weighted_huber/`
- `experiments/pair_policy/runs/train/segformer_mit_b4_minmax_bce/`
- `experiments/pair_policy/runs/train/segformer_mit_b4_exp_kl_tau01/`
- `experiments/pair_policy/runs/train/unetpp_resnet50_minmax_weighted_huber/`

Each run writes:

- `console.log`
- `last.pt`
- `best.pt`
- `best.json`
- `qualitative.png`

## Monitoring

Example:

```bash
tail -f experiments/pair_policy/runs/train/segformer_mit_b4_minmax_weighted_huber/console.log
```

## Legacy Configs

Older probes and ablations were moved to:

- `configs/train/archive_legacy/`

These are kept for reference, but they are no longer considered active comparison baselines.
