"""SuperPoint local features into a COLMAP database, SALAD global descriptors
into a sidecar file beside it (see doc/panorama_mapping.md).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pycolmap
from pycolmap import logging

from .progress import map_with_progress

GLOBAL_FEATURES_FILENAME = "global_features.npz"

# Images per progress log line.
_LOG_EVERY = 200

# COLMAP's descriptors table is typed uint8, and float descriptors reach it by
# byte reinterpretation under one of the learned extractor types -- the only
# lossless way to store SuperPoint's 256 floats per keypoint in the schema.
# The tag is storage, not truth: COLMAP's own matchers dispatch on it, so this
# database is for this pipeline's mapper, not for `colmap matcher`.
_FLOAT_DESCRIPTOR_TYPE = pycolmap.FeatureExtractorType.ALIKED_N32


def _keypoints_blob(keypoints):
    """COLMAP's Nx4 keypoints layout (x, y, scale, orientation). SuperPoint has
    no affine shape, so scale and orientation stay at zero."""
    blob = np.zeros((len(keypoints), 4), dtype=np.float32)
    blob[:, :2] = keypoints
    return blob


def extract_features(workspace_path, local_extractor, global_extractor, num_threads=8):
    """Extracts features for every image in `<workspace_path>/database.db`.

    Writes each image's keypoints and descriptors into the database, strongest
    detection first, and every global descriptor into the returned sidecar
    file. The sidecar is written last, so its presence means the stage
    finished; `run_pipeline` skips the stage on that basis.

    Inference runs on `num_threads` worker threads; the database writes stay on
    this one.
    """
    workspace_path = Path(workspace_path)
    images_dir = workspace_path / "images"
    global_features_path = workspace_path / GLOBAL_FEATURES_FILENAME
    logging.info(f"Extraction settings: num_threads={num_threads}")

    image_ids = []
    global_features = []

    with pycolmap.Database.open(workspace_path / "database.db") as database:
        images = database.read_all_images()
        # Every image is re-extracted, so drop whatever a previous run left:
        # writing a second row for an image the tables already hold would fail.
        database.clear_keypoints()
        database.clear_descriptors()

        def extract(image):
            image_path = images_dir / image.name
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read {image_path}")
            return (
                image.image_id,
                local_extractor.extract(image_bgr),
                global_extractor.extract(image_bgr),
            )

        for image_id, local, global_feature in map_with_progress(
            extract, images, "Extracted features for", num_threads, _LOG_EVERY
        ):
            database.write_keypoints(image_id, _keypoints_blob(local.keypoints))
            database.write_descriptors(
                image_id,
                pycolmap.FeatureDescriptors.from_float(
                    pycolmap.FeatureDescriptorsFloat(
                        _FLOAT_DESCRIPTOR_TYPE, local.descriptors
                    )
                ),
            )
            image_ids.append(image_id)
            global_features.append(global_feature)

    np.savez(
        global_features_path,
        image_ids=np.array(image_ids, dtype=np.int64),
        features=np.stack(global_features),
    )
    logging.info(
        f"Extracted features for {len(image_ids)} images; wrote global "
        f"descriptors to {global_features_path}"
    )
    return global_features_path


def read_global_features(global_features_path):
    """The `image_id -> descriptor` map `extract_features` buffered."""
    with np.load(global_features_path) as data:
        return dict(zip(data["image_ids"].tolist(), data["features"], strict=True))
