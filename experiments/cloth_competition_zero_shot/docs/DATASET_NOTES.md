# Cloth Competition Dataset Notes

This note tracks the external dataset used for zero-shot evaluation.

## Dataset Identity

Paper title:

- `A Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection: The ICRA 2024 Cloth Competition`

Current public source:

- competition page: `https://airo.ugent.be/cloth_competition/`
- Zenodo record: `https://zenodo.org/records/14621179`
- DOI: `https://doi.org/10.5281/zenodo.14621179`

Current local download target:

- `${DATASET_ROOT}/cloth_competition/ICRA_2024_cloth_competition_dataset.zip`

Reference metadata recorded during download setup:

- file name: `ICRA_2024_cloth_competition_dataset.zip`
- size: `82,978,415,452` bytes
- license: `CC BY 4.0`
- MD5: `74b5a659d455c5cc4c7898db126ea6ef`

Related local note:

- `${DATASET_ROOT}/cloth_competition/ICRA_2024_cloth_competition_dataset_SOURCE.md`

## Verified From Zip Inspection

The dataset has already been inspected directly from the public zip archive.

Observed structure:

- top-level sample groups such as `cloth_competition_dataset_0000/`
- per-sample folders such as `sample_000000/`
- inside each sample:
  - `observation_start/`
  - `grasp/`
  - `observation_result/`
  - `episode.mp4`

Observed modalities:

- frontal RGB:
  - `observation_start/image_left.png`
  - `observation_result/image_left.png`
- right RGB:
  - `observation_start/image_right.png`
  - `observation_result/image_right.png`
- depth:
  - `depth_image.jpg`
  - `depth_map.tiff`
- confidence:
  - `confidence_map.tiff`
- point cloud:
  - `point_cloud.ply`
- camera and robot metadata:
  - intrinsics
  - camera pose
  - arm pose / joints
  - tcp poses

Observed grasp annotation files:

- `grasp/grasp_annotation.json`
- `grasp/grasp_pose.json`
- `grasp/frontal_image_grasp.jpg`
- `grasp/topdown_image_grasp.jpg`

Important verified fact:

- the benchmark exposes a **single grasp point**, not an ordered two-grasp pair

Observed `grasp_annotation.json` fields:

- `clicked_point_frontal`
- `clicked_point_topdown`
- `grasp_depth`

Observed `grasp_pose.json` fields:

- `position_in_meters`
- `rotation_euler_xyz_in_radians`

Current observed sample count from zip enumeration used by the benchmark script:

- `503`

Immediate implication for our project:

- this dataset is suitable for a **single-grasp zero-shot proxy evaluation**
- it is **not** a direct pair-policy benchmark unless we derive an additional mapping from our pair semantics to the single-grasp protocol

## Benchmark Integration Questions

These are the concrete questions that matter for our `A1/A2` model.

1. Can each benchmark sample be mapped to a single RGB image input for our model?
2. Is there a cloth foreground mask available, or do we need to derive one?
3. Can the single-grasp annotation be used as a reasonable proxy for one peak of `A1/A2`?
4. Are grasps defined in:
   - image coordinates
   - normalized image coordinates
   - 3D/world coordinates
5. Is the benchmark action semantics close enough to our current dual-grasp action parameterization?
6. Are there benchmark success scores or rankings that can support a stronger evaluation than point distance?

## Integration Risks

- benchmark grasp semantics differ from our ordered-pair assumption
- camera viewpoint and crop convention may differ from current pair-policy training
- the benchmark may use real images with clutter/background conditions outside our synthetic render domain
- the benchmark may not expose enough metadata for direct closed-loop replay

## Initial Practical Plan

Use the benchmark in increasing order of difficulty:

1. single-grasp proxy evaluation with image-space point distance
2. stronger baselines and better cropping / segmentation
3. investigate whether `grasp_pose.json` can support a richer geometric mapping
4. simulator replay only if mapping quality is acceptable
