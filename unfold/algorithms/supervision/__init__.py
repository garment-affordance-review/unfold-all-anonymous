from .teacher_pointcept import TeacherRewardInfer
from .projection import (
    best_candidate_xy,
    build_dense_a1_heatmap,
    build_dense_a2_heatmap_from_row,
    compute_reward_row_margin,
    compute_top1_margin,
    normalize_finite_scores,
    row_softmax_masked,
)
from .targets import (
    build_reward_matrix,
    build_a1_from_reward_matrix,
    build_a2_conditional_topk,
)

__all__ = [
    "TeacherRewardInfer",
    "build_reward_matrix",
    "build_a1_from_reward_matrix",
    "build_a2_conditional_topk",
    "best_candidate_xy",
    "build_dense_a1_heatmap",
    "build_dense_a2_heatmap_from_row",
    "compute_reward_row_margin",
    "compute_top1_margin",
    "normalize_finite_scores",
    "row_softmax_masked",
]
