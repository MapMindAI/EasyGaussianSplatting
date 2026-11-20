"""Sky probability from xiongzhu666/Sky-Segmentation-and-Post-processing's
U²-Net export, run locally with ONNX Runtime.

The graph takes ImageNet-normalized RGB shaped (1, 3, 320, 320) and returns
seven sigmoid maps (1, 1, 320, 320); the first is U²-Net's fused output.
"""

from pathlib import Path

import cv2
import numpy as np

from mapping.onnx_session import onnx_session

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "skyseg.onnx"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Sky probability above which a pixel is sky.
SKY_THRESHOLD = 0.5


def model_input(image_bgr, height, width):
    """(1, 3, height, width) float32 ImageNet-normalized RGB."""
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[np.newaxis]


class SkySegmenter:
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self._session = onnx_session(model_path)
        model_input_info = self._session.get_inputs()[0]
        self._input_name = model_input_info.name
        self._height, self._width = model_input_info.shape[-2:]
        self._output_name = self._session.get_outputs()[0].name

    def probability(self, image_bgr):
        """(H, W) float32 sky probability in [0, 1] at the image's own size.

        The image is resized to the model's square input, ignoring aspect ratio.
        """
        (probability,) = self._session.run(
            [self._output_name],
            {self._input_name: model_input(image_bgr, self._height, self._width)},
        )
        return cv2.resize(
            probability[0, 0], (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    def mask(self, image_bgr):
        """(H, W) bool, true where the sky probability is above SKY_THRESHOLD."""
        return self.probability(image_bgr) > SKY_THRESHOLD
