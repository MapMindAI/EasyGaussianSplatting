"""VGGT-Omega depth from an exported ONNX model, run locally with ONNX Runtime.

The graph takes `images`, float32 RGB in [0, 1] shaped (1, N, 3, H, W) at a
fixed H and W, and returns `depth` (1, N, H, W, 1), `depth_conf` (1, N, H, W),
and `pose_enc` (1, N, 9); see the `.json` exported beside the model.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

from mapping.onnx_session import onnx_session

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "vggt_omega_416.onnx"
# VGGT confidence starts at 1 and runs to tens; sky and far background sit
# below 2.5, so this puts that cut at the shared --min-confidence default of 0.5.
CONFIDENCE_SCALE = 1 / 5


def quaternion_to_matrix(quaternion):
    """The rotation for an (x, y, z, w) quaternion."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def decode_pose(pose_encoding, height, width):
    """(3, 4) OpenCV camera-from-world and (3, 3) intrinsics for an image of
    `height` x `width`, from VGGT-Omega's (tx, ty, tz, qx, qy, qz, qw,
    vertical fov, horizontal fov) encoding."""
    translation, quaternion = pose_encoding[:3], pose_encoding[3:7]
    vertical_fov, horizontal_fov = pose_encoding[7], pose_encoding[8]
    camera_from_world = np.column_stack([quaternion_to_matrix(quaternion), translation])
    intrinsics = np.array([
        [width / 2 / np.tan(horizontal_fov / 2), 0, width / 2],
        [0, height / 2 / np.tan(vertical_fov / 2), height / 2],
        [0, 0, 1],
    ])
    return camera_from_world, intrinsics


class VggtOmegaDepthEstimator:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        import onnxruntime

        options = onnxruntime.SessionOptions()
        # The export's unused initializers otherwise log a warning each.
        options.log_severity_level = 3
        self._session = onnx_session(model_path, options)
        if self._session.get_providers()[0] == "CUDAExecutionProvider":
            device_id = self._session.get_provider_options()["CUDAExecutionProvider"]["device_id"]
            arena = f"gpu:{device_id}"
        else:
            arena = "cpu:0"
        # Group sizes vary, so the memory arena would otherwise keep every
        # differently sized buffer it ever grew for until the GPU runs out.
        # Shrinking returns its unused regions after each run.
        self._run_options = onnxruntime.RunOptions()
        self._run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", arena)
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._height, self._width = model_input.shape[-2:]
        # A 12-view group's attention alone takes 4.4 GB, so callers on several
        # threads take turns on the GPU rather than each allocating their own.
        self._run_lock = threading.Lock()

    def run(self, images_bgr):
        """Per input image: `depth_list` and `depth_conf_list` (H, W) maps,
        `camera_from_world_list` (3, 4), and `intrinsics_list` (3, 3).

        Confidence is the model's, times CONFIDENCE_SCALE. Each image is resized to the model's input, ignoring aspect ratio, and
        its maps and intrinsics are scaled back to the image's own resolution.
        Depths and translations share one arbitrary scale across the call, in
        a world frame of the model's choosing.
        """
        frames = np.stack([
            cv2.cvtColor(
                cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            ).transpose(2, 0, 1)
            for image in images_bgr
        ]).astype(np.float32) / 255.0

        with self._run_lock:
            depth, confidence, pose_encodings = self._session.run(
                ["depth", "depth_conf", "pose_enc"], {self._input_name: frames[None]},
                self._run_options,
            )
        cameras = [
            decode_pose(pose_encoding, *image.shape[:2])
            for pose_encoding, image in zip(pose_encodings[0], images_bgr)
        ]

        def at_source_size(maps):
            return [
                cv2.resize(value, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                for value, image in zip(maps.reshape(len(images_bgr), self._height, self._width), images_bgr)
            ]

        return {
            "depth_list": at_source_size(depth),
            "depth_conf_list": at_source_size(confidence * CONFIDENCE_SCALE),
            "camera_from_world_list": [camera_from_world for camera_from_world, _ in cameras],
            "intrinsics_list": [intrinsics for _, intrinsics in cameras],
        }
