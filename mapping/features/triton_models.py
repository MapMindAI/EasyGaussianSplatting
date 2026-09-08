"""SuperPoint, LightGlue, and SALAD clients backed by a Triton server.

Thin adapters over the clients vendored in
`third_party/EasyTensorRT/triton_client`, which import each other by bare
module name, so that directory goes on `sys.path` instead of being imported as
a package. The adapters normalize away the batch axis and the fixed-size
keypoint blocks the exported graphs use, so callers only ever see the
detections an image actually has.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TRITON_CLIENT_DIR = (
    Path(__file__).resolve().parents[2] / "third_party" / "EasyTensorRT" / "triton_client"
)
if not (_TRITON_CLIENT_DIR / "superpoint.py").is_file():
    raise ImportError(
        f"Missing {_TRITON_CLIENT_DIR}; run: "
        "git submodule update --init third_party/EasyTensorRT"
    )
if str(_TRITON_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(_TRITON_CLIENT_DIR))

from lightglue import LightGlue  # noqa: E402
from salad import SaladClient  # noqa: E402
from superpoint import SuperPoint  # noqa: E402

# LightGlue's exported graph takes a fixed-size keypoint block per image (see
# third_party/EasyTensorRT/model_repository*/lightglue_*/config.pbtxt). Nothing
# downstream can use a detection past that -- the mapper only ever sees
# keypoints LightGlue matched -- so extraction caps at it rather than
# persisting keypoints no stage would read. Re-exporting the graph larger means
# re-extracting.
MATCHER_NUM_KEYPOINTS = 512

# Longest side SuperPoint detects on. A larger image is downscaled to it and
# its keypoints scaled back, so a cube face of any size still lands in
# full-resolution coordinates. It is the size the TensorRT plan is built
# around, and detecting on a 2000-pixel face instead costs several times the
# GPU memory for detections the 512-point matcher input cannot carry anyway.
DETECTOR_MAX_IMAGE_SIZE = 960


@dataclass
class LocalFeatures:
    """One image's detections, strongest first."""

    keypoints: np.ndarray  # (N, 2) in image coordinates
    descriptors: np.ndarray  # (N, D) float32


def _without_batch_axis(array, ndim):
    array = np.asarray(array, dtype=np.float32)
    return array[0] if array.ndim > ndim else array


class LocalFeatureExtractor:
    """SuperPoint keypoints and descriptors."""

    def __init__(self, triton_url, keypoint_threshold=0.015, model_version="1"):
        self._client = SuperPoint(
            triton_url,
            model_version=model_version,
            max_image_shape=DETECTOR_MAX_IMAGE_SIZE,
            keypoint_thresh=keypoint_threshold,
        )

    def extract(self, image_bgr) -> LocalFeatures:
        keypoints, descriptors, scores = self._client.run(image_bgr)
        keypoints = _without_batch_axis(keypoints, 2)
        descriptors = _without_batch_axis(descriptors, 2)
        scores = _without_batch_axis(scores, 1)
        # The TensorRT plan returns a fixed-size block padded with zero-score
        # entries while the ONNX model returns only its detections; keeping the
        # positive scores covers both. A real detection always scores at least
        # the extractor's threshold.
        strongest = np.argsort(-scores)[:MATCHER_NUM_KEYPOINTS]
        strongest = strongest[scores[strongest] > 0.0]
        return LocalFeatures(keypoints[strongest], descriptors[strongest])


class GlobalFeatureExtractor:
    """One SALAD place-recognition descriptor per image."""

    def __init__(self, triton_url, model_version="1"):
        self._client = SaladClient(triton_url=triton_url, model_version=model_version)

    def extract(self, image_bgr) -> np.ndarray:
        return np.asarray(self._client.run(image_bgr), dtype=np.float32).reshape(-1)


class FeatureMatcher:
    """LightGlue matches between two images' local features."""

    def __init__(self, triton_url, match_threshold=0.2, model_version="1"):
        self._client = LightGlue(
            triton_url, model_version=model_version, match_thresh=match_threshold
        )

    def match(self, features1, image_size1, features2, image_size2) -> np.ndarray:
        """Matches two images' features, each `image_size` a (width, height).

        Only each image's leading `MATCHER_NUM_KEYPOINTS` detections take part.
        `LocalFeatures` is ordered strongest first, so that prefix is the
        strongest subset and the indices need no remapping.

        Returns the (M, 2) uint32 index pairs into `features1.keypoints` and
        `features2.keypoints` that COLMAP's `matches` table expects.
        """
        keypoints1, descriptors1, mask1 = _padded_block(features1)
        keypoints2, descriptors2, mask2 = _padded_block(features2)
        match_indices, _ = self._client.run(
            keypoints1,
            descriptors1,
            _image_shape(image_size1),
            keypoints2,
            descriptors2,
            _image_shape(image_size2),
            mask0=mask1,
            mask1=mask2,
        )
        num_keypoints1 = min(len(features1.keypoints), MATCHER_NUM_KEYPOINTS)
        num_keypoints2 = min(len(features2.keypoints), MATCHER_NUM_KEYPOINTS)
        matched = np.asarray(match_indices).reshape(-1)[:num_keypoints1]
        # Padded rows match nothing, but bound the result anyway: an index past
        # either image's detections would corrupt the database.
        indices1 = np.flatnonzero((matched >= 0) & (matched < num_keypoints2))
        return np.column_stack([indices1, matched[indices1]]).astype(np.uint32)


def _padded_block(features):
    """The (keypoints, descriptors, mask) triple LightGlue's fixed-size inputs
    expect, holding the strongest detections that fit, with the unfilled rows
    masked off."""
    strongest_keypoints = features.keypoints[:MATCHER_NUM_KEYPOINTS]
    strongest_descriptors = features.descriptors[:MATCHER_NUM_KEYPOINTS]
    count = len(strongest_keypoints)
    keypoints = np.zeros((1, MATCHER_NUM_KEYPOINTS, 2), dtype=np.float32)
    descriptors = np.zeros(
        (1, MATCHER_NUM_KEYPOINTS, features.descriptors.shape[1]), dtype=np.float32
    )
    mask = np.zeros((1, MATCHER_NUM_KEYPOINTS, 1), dtype=bool)
    keypoints[0, :count] = strongest_keypoints
    descriptors[0, :count] = strongest_descriptors
    mask[0, :count] = True
    return keypoints, descriptors, mask


def _image_shape(image_size):
    width, height = image_size
    return np.array([[height, width]], dtype=np.int32)
