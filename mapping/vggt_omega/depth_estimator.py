"""VGGT-Omega depth from an exported ONNX model, run locally with ONNX Runtime.

The graph takes `images`, float32 RGB in [0, 1] shaped (1, N, 3, H, W) at a
fixed H and W, and returns `depth` (1, N, H, W, 1) and `depth_conf`
(1, N, H, W); see the `.json` exported beside the model.
"""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "vggt_omega_416.onnx"


class VggtOmegaDepthEstimator:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        import onnxruntime

        if not model_path.is_file():
            raise FileNotFoundError(f"No VGGT-Omega ONNX model at {model_path}")
        # onnxruntime-gpu ships without cuDNN; load the one torch's nvidia wheels carry.
        onnxruntime.preload_dlls()
        options = onnxruntime.SessionOptions()
        # The export's unused initializers otherwise log a warning each.
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        if self._session.get_providers()[0] != "CUDAExecutionProvider":
            print(f"CUDA unavailable to ONNX Runtime; running {model_path.name} on the CPU")
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        self._height, self._width = model_input.shape[-2:]

    def run(self, images_bgr):
        """`depth_list` and `depth_conf_list`, one (H, W) array per input image.

        Each image is resized to the model's input, ignoring aspect ratio, and
        its maps are resized back to the image's own resolution. The depths
        share one arbitrary scale across the call; the predicted cameras are
        dropped, since callers already have the COLMAP poses.
        """
        frames = np.stack([
            cv2.cvtColor(
                cv2.resize(image, (self._width, self._height), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2RGB,
            ).transpose(2, 0, 1)
            for image in images_bgr
        ]).astype(np.float32) / 255.0

        depth, confidence = self._session.run(
            ["depth", "depth_conf"], {self._input_name: frames[None]}
        )

        def at_source_size(maps):
            return [
                cv2.resize(value, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                for value, image in zip(maps.reshape(len(images_bgr), self._height, self._width), images_bgr)
            ]

        return {
            "depth_list": at_source_size(depth),
            "depth_conf_list": at_source_size(confidence),
        }
