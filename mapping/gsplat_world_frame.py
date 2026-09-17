"""gsplat's world normalization: the scale it applies, and undoing it on export.

gsplat's COLMAP parser trains in a similarity-normalized frame (see
`third_party/gsplat/examples/datasets/colmap.py`), but `simple_trainer` exports
the raw parameters, so the saved point cloud lands in that frame rather than the
input model's. The inverse similarity returns the means to the COLMAP frame and
carries the Gaussian orientations and their (log) sizes along with it. The
forward direction is here too, for the depth supervision that arrives in the
COLMAP frame and has to reach the one the renderer measures in.
"""
import math

import numpy as np


def _rotation_quaternion(matrix):
    """The wxyz quaternion of a 3x3 rotation matrix."""
    # Each branch solves for the component the matrix makes largest, so the
    # divisor `s` stays far from zero; picking one fixed branch loses precision
    # near the rotations where its component vanishes.
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * s,
            (matrix[2, 1] - matrix[1, 2]) / s,
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ]
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quaternion = [
            (matrix[2, 1] - matrix[1, 2]) / s,
            0.25 * s,
            (matrix[0, 1] + matrix[1, 0]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
        ]
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quaternion = [
            (matrix[0, 2] - matrix[2, 0]) / s,
            (matrix[0, 1] + matrix[1, 0]) / s,
            0.25 * s,
            (matrix[1, 2] + matrix[2, 1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quaternion = [
            (matrix[1, 0] - matrix[0, 1]) / s,
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
        ]
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_product(left, right):
    """Hamilton product of wxyz quaternions, which broadcasts `left` over `right`."""
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
            left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
            left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
            left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        ],
        axis=-1,
    )


def similarity_scale(transform):
    """The uniform scale a similarity applies.

    A similarity's linear part is `scale * rotation`, so the determinant
    recovers the scale and stripping it leaves a pure rotation.
    """
    return float(abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0))


def restore_colmap_coordinates(transform, means, scales, quats):
    """Map Gaussian parameters from gsplat's normalized frame back to COLMAP's.

    `transform` is the parser's `transform`, the similarity taking the COLMAP
    frame to the normalized one. `means` (N, 3) and `quats` (N, 4, wxyz) are the
    exported arrays and `scales` (N, 3) the log-scales the PLY stores.
    """
    inverse = np.linalg.inv(transform)
    scale = similarity_scale(transform)
    # The inverse scales by 1/scale, so multiplying it back through leaves the
    # pure rotation a quaternion can represent.
    orientation = _rotation_quaternion(inverse[:3, :3] * scale)
    means = np.asarray(means) @ inverse[:3, :3].T + inverse[:3, 3]
    # Dividing the sizes by `scale` is a subtraction in the log domain.
    scales = np.asarray(scales) - math.log(scale)
    quats = _quaternion_product(orientation, np.asarray(quats))
    return means, scales, quats


def depth_supervision_in_normalized_frame(transform, rows, width, height):
    """Put generate_depth.py's rows into the frame the renderer works in.

    `rows` is its (M, 3) of x and y normalized to [0, 1] and depth in the COLMAP
    frame; `transform` is the parser's COLMAP-to-normalized similarity. Returns
    the (M, 2) pixel coordinates of the image as loaded and the (M,) depths, and
    a camera carried through that similarity sees depths multiplied by its
    scale, so that is the direction here.
    """
    rows = np.asarray(rows, dtype=np.float32)
    points = rows[:, :2] * np.array([width - 1, height - 1], dtype=np.float32)
    depths = rows[:, 2] * (1.0 / similarity_scale(transform))
    return points, depths.astype(np.float32)


_scene_transform = None


def capture_scene_transform(parser_class):
    """Remember each parser's COLMAP-to-normalized transform for the exporter."""
    original_init = parser_class.__init__

    def __init__(self, *args, **kwargs):
        global _scene_transform
        original_init(self, *args, **kwargs)
        # `self.transform` only exists once the original __init__ has solved it.
        _scene_transform = self.transform

    parser_class.__init__ = __init__


def export_in_colmap_frame(exporter_module):
    """Make `export_splats` write point clouds in the input COLMAP frame.

    `simple_trainer` imports `export_splats` from `gsplat`, so replacing the
    attribute reaches it; the transform `capture_scene_transform` recorded is
    undone on every export. `simple_trainer` copies the name into its own
    namespace as it is imported, so this has to run before the trainer does.
    """
    original_export = exporter_module.export_splats

    def export_splats(*args, **kwargs):
        # No parser was built, so there is no normalization to undo.
        if _scene_transform is not None:
            # Opacities and the spherical harmonics survive a similarity
            # unchanged, so only these three need moving. The trainer passes
            # them by keyword.
            means, scales, quats = kwargs["means"], kwargs["scales"], kwargs["quats"]
            restored = restore_colmap_coordinates(
                _scene_transform,
                means.detach().cpu().numpy(),
                scales.detach().cpu().numpy(),
                quats.detach().cpu().numpy(),
            )
            # Rebinding the arguments leaves the trainer's own parameters in the
            # normalized frame it keeps optimizing in; `new_tensor` carries the
            # originals' device and dtype over to the copies that get written.
            kwargs["means"] = means.new_tensor(restored[0])
            kwargs["scales"] = scales.new_tensor(restored[1])
            kwargs["quats"] = quats.new_tensor(restored[2])
        return original_export(*args, **kwargs)

    exporter_module.export_splats = export_splats
