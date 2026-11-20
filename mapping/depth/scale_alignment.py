"""One depth scale per multi-view group, from sparse depths and the model's cameras.

Ported from ArSession's dense_recon/sparse_depth_alignment. A model that also
predicts cameras gives a first scale from its camera baselines against the
reference ones; the sparse COLMAP depths the group observes then refine it
with a robust, frame-balanced fit. Either source alone can fix the scale.
"""

from dataclasses import dataclass

import numpy as np

from mapping.depth.covisibility_groups import MIN_BASELINE, centre_and_rotation, rotation_angle_degrees

MIN_TRAINING_PAIRS = 30
MIN_TRAINING_FRAMES = 2
MIN_INLIER_RATIO = 0.3
IRLS_ITERATIONS = 8
HUBER_CONSTANT = 1.345
# Sparse references beyond this depth, in map units, are too noisy to fit to.
MAX_REFERENCE_DEPTH = 20.0
MAX_POSE_ANGLE_DEGREES = 5.0


@dataclass(frozen=True)
class PoseScale:
    scale: float
    median_error: float  # Median normalized baseline error at that scale.


def weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def usable_depths(predicted, reference):
    return (
        np.isfinite(predicted) & np.isfinite(reference)
        & (predicted > 0) & (reference > 0) & (reference <= MAX_REFERENCE_DEPTH)
    )


def pose_depth_scale(predicted_camera_from_world, reference_camera_from_world):
    """The scale carrying the model's camera baselines onto the reference ones,
    or None when too few pairs agree. Inputs are (3, 4) per frame."""
    predicted = [centre_and_rotation(pose) for pose in predicted_camera_from_world]
    reference = [centre_and_rotation(pose) for pose in reference_camera_from_world]

    def relative(cameras, first, second):
        """The second camera's rotation and offset seen from the first."""
        (first_centre, first_rotation), (second_centre, second_rotation) = (
            cameras[first], cameras[second]
        )
        return first_rotation.T @ second_rotation, first_rotation.T @ (second_centre - first_centre)

    candidates, pairs = 0, []
    for first in range(len(reference)):
        for second in range(first + 1, len(reference)):
            reference_rotation, reference_offset = relative(reference, first, second)
            predicted_rotation, predicted_offset = relative(predicted, first, second)
            length = np.linalg.norm(reference_offset)
            if not np.isfinite(length) or length < MIN_BASELINE:
                continue
            candidates += 1
            predicted_length = np.linalg.norm(predicted_offset)
            if (
                not np.isfinite(predicted_length)
                or predicted_length < 1e-6
                or rotation_angle_degrees(predicted_rotation.T @ reference_rotation)
                > MAX_POSE_ANGLE_DEGREES
                or predicted_offset @ reference_offset / (predicted_length * length)
                < np.cos(np.radians(MAX_POSE_ANGLE_DEGREES))
            ):
                continue
            pairs.append((
                first,
                second,
                predicted_offset / length,
                reference_offset / length,
                np.log(length / predicted_length),
            ))
    if not pairs:
        return None

    centre = float(np.median([log_scale for *_, log_scale in pairs]))
    inliers = [
        (first, second, predicted_offset, reference_offset, log_scale)
        for first, second, predicted_offset, reference_offset, log_scale in pairs
        if abs(log_scale - centre) <= 0.15
        and np.linalg.norm(np.exp(centre) * predicted_offset - reference_offset) <= 0.25
    ]
    log_scales_by_frame = {}
    for first, second, _, _, log_scale in inliers:
        log_scales_by_frame.setdefault(first, []).append(log_scale)
        log_scales_by_frame.setdefault(second, []).append(log_scale)
    if len(log_scales_by_frame) < 3 or len(inliers) < 2 or len(inliers) < 0.6 * candidates:
        return None

    predicted_offsets = np.array([predicted_offset for _, _, predicted_offset, _, _ in inliers])
    reference_offsets = np.array([reference_offset for _, _, _, reference_offset, _ in inliers])
    scale = float((predicted_offsets * reference_offsets).sum() / (predicted_offsets**2).sum())
    errors = np.sort(np.linalg.norm(scale * predicted_offsets - reference_offsets, axis=1))
    frame_medians = [np.median(values) for values in log_scales_by_frame.values()]
    median_error = float(np.median(errors))
    if (
        not np.isfinite(scale)
        or scale <= 0
        or median_error > 0.10
        or errors[int(np.ceil(len(errors) * 0.9)) - 1] > 0.25
        or max(frame_medians) - min(frame_medians) > 0.15
    ):
        return None
    return PoseScale(scale, median_error)


def align_sparse_depth(frame_indices, point_ids, predicted, reference, pose_scale=None):
    """The scale carrying `predicted` depths onto `reference` ones, or None.

    Each sample is one sparse point observed by one group frame. Frames weigh
    equally however many points they see; the fit starts at the weighted
    median log ratio, keeps samples within a MAD band of it, and refines with
    Huber reweighting. A `pose_scale` joins as one more robust term, which also
    lets a group with too few sparse points be scaled.
    """
    frame_indices, point_ids = np.asarray(frame_indices), np.asarray(point_ids)
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    usable = usable_depths(predicted, reference)
    frame_indices, point_ids = frame_indices[usable], point_ids[usable]
    predicted, reference = predicted[usable], reference[usable]
    frames, frame_positions, frame_counts = np.unique(
        frame_indices, return_inverse=True, return_counts=True
    )
    if pose_scale is None and (
        len(predicted) < MIN_TRAINING_PAIRS or len(frames) < MIN_TRAINING_FRAMES
    ):
        return None

    weights = 1.0 / frame_counts[frame_positions]
    log_ratios = np.log(reference / predicted)
    if len(log_ratios):
        centre = weighted_median(log_ratios, weights)
        spread = weighted_median(np.abs(log_ratios - centre), weights)
        threshold = float(np.clip(3 * 1.4826 * spread, 0.05, 0.35))
    else:
        centre, threshold = np.log(pose_scale.scale), 0.05
    inliers = np.abs(log_ratios - centre) <= threshold
    if pose_scale is None and (
        inliers.sum() < MIN_TRAINING_PAIRS
        or len(np.unique(frame_indices[inliers])) < MIN_TRAINING_FRAMES
        or inliers.mean() < MIN_INLIER_RATIO
    ):
        return None
    if pose_scale is not None and len(frames):
        # Sparse evidence weighs at most as much as the pose term, less when thin.
        weights = weights * min(1.0, len(np.unique(point_ids)) / MIN_TRAINING_PAIRS) / len(frames)

    normalized = predicted[inliers] / reference[inliers]
    weights = weights[inliers]
    scale = pose_scale.scale if pose_scale is not None else float(np.exp(centre))
    if pose_scale is not None:
        pose_normalized = 1 / pose_scale.scale
        pose_sigma = max(0.02, pose_scale.median_error)
    for _ in range(IRLS_ITERATIONS):
        residuals = scale * normalized - 1
        sigma = 0.02
        if len(residuals):
            sigma = max(sigma, 1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))))
        huber = np.minimum(1.0, HUBER_CONSTANT * sigma / np.maximum(np.abs(residuals), 1e-12))
        numerator = float((weights * huber * normalized).sum())
        denominator = float((weights * huber * normalized**2).sum())
        if pose_scale is not None:
            pose_residual = scale * pose_normalized - 1
            pose_weight = min(1.0, HUBER_CONSTANT * pose_sigma / max(abs(pose_residual), 1e-12))
            numerator += pose_weight * pose_normalized
            denominator += pose_weight * pose_normalized**2
        scale = numerator / denominator
        if not np.isfinite(scale) or scale <= 0:
            return None
    return float(scale)


def median_relative_error(scale, predicted, reference):
    """How far one frame's scaled depths sit from its sparse ones; inf if unmeasured."""
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    usable = usable_depths(predicted, reference)
    if not usable.any():
        return np.inf
    return float(np.median(np.abs(scale * predicted[usable] / reference[usable] - 1)))
