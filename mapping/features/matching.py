"""Pair selection and LightGlue matching into a COLMAP database.

Sequential pairs cover each cube face's own neighbourhood in capture order;
retrieval pairs cover revisits from elsewhere in the capture. A pair selected
by either is matched. See doc/panorama_mapping.md for why the faces are matched
only against themselves.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pycolmap
from pycolmap import logging

from .extraction import GLOBAL_FEATURES_FILENAME, read_global_features
from .progress import map_with_progress
from .triton_models import LocalFeatures

# Image pairs per progress log line.
_LOG_EVERY = 1000

# Images whose features are kept in memory while matching. Each image appears
# in tens of pairs, so re-reading its blobs per pair would dominate the stage;
# at 512 keypoints x 256 float32 dims this bounds the cache to about 270 MB.
_FEATURE_CACHE_SIZE = 512


@dataclass
class PairSelection:
    num_sequential: int = 10
    num_retrieval: int = 10
    num_retrieval_excluded: int = 50


def _image_sequences(database):
    """Each camera's image ids in capture order, keyed by camera id. One camera
    is one cube face, and frame ids increase with capture order."""
    images = {image.data_id: image for image in database.read_all_images()}
    sequences = defaultdict(list)
    for frame in sorted(database.read_all_frames(), key=lambda frame: frame.frame_id):
        for data_id in frame.data_ids:
            image = images[data_id]
            sequences[image.camera_id].append(image.image_id)
    return sequences


def _sequential_pairs(sequences, num_sequential):
    pairs = set()
    for sequence in sequences.values():
        for index, image_id in enumerate(sequence):
            for other_id in sequence[index + 1 : index + 1 + num_sequential]:
                pairs.add((min(image_id, other_id), max(image_id, other_id)))
    return pairs


def _retrieval_pairs(sequences, global_features, selection):
    pairs = set()
    if selection.num_retrieval <= 0 or not global_features:
        return pairs

    image_ids = sorted(global_features)
    descriptors = np.stack([global_features[image_id] for image_id in image_ids])
    descriptors /= np.clip(
        np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12, None
    )
    similarity = descriptors @ descriptors.T

    camera_of, index_of = {}, {}
    for camera_id, sequence in sequences.items():
        for index, image_id in enumerate(sequence):
            camera_of[image_id] = camera_id
            index_of[image_id] = index
    cameras = np.array([camera_of[image_id] for image_id in image_ids])
    indices = np.array([index_of[image_id] for image_id in image_ids])

    for row, image_id in enumerate(image_ids):
        excluded = (cameras == cameras[row]) & (
            np.abs(indices - indices[row]) <= selection.num_retrieval_excluded
        )
        scores = np.where(excluded, -np.inf, similarity[row])
        nearest = np.argpartition(-scores, min(selection.num_retrieval, len(scores) - 1))
        for other_row in nearest[: selection.num_retrieval]:
            if np.isfinite(scores[other_row]):
                other_id = image_ids[other_row]
                pairs.add((min(image_id, other_id), max(image_id, other_id)))
    return pairs


def select_pairs(database, global_features, selection):
    """The candidate image pairs to match, as sorted (image_id1, image_id2)."""
    sequences = _image_sequences(database)
    sequential = _sequential_pairs(sequences, selection.num_sequential)
    retrieval = _retrieval_pairs(sequences, global_features, selection)
    pairs = sequential | retrieval
    logging.info(
        f"Selected {len(pairs)} pairs: {len(sequential)} sequential, "
        f"{len(retrieval - sequential)} retrieval-only"
    )
    return sorted(pairs)


def match_features(workspace_path, matcher, selection, num_threads=8):
    """Matches the selected pairs of `<workspace_path>/database.db` and writes
    each pair's matches and verified two-view geometry back into it.

    Pairs the database already holds a two-view geometry for are left alone, so
    an interrupted run resumes rather than restarting. Matching and
    verification run on `num_threads` worker threads; the database is touched
    under a lock, since COLMAP opens SQLite without one.

    Returns the number of pairs written.
    """
    workspace_path = Path(workspace_path)
    two_view_geometry_options = pycolmap.TwoViewGeometryOptions()
    logging.info(f"Matching settings: num_threads={num_threads}, {selection}")

    database_lock = threading.Lock()
    num_pairs_written = 0
    with pycolmap.Database.open(workspace_path / "database.db") as database:
        selected = select_pairs(
            database,
            read_global_features(workspace_path / GLOBAL_FEATURES_FILENAME),
            selection,
        )
        # Per pair rather than one bulk read: read_two_view_geometry_num_inliers
        # reports only the pairs whose geometry has a usable configuration, so
        # it would miss rows here and the re-write would hit a UNIQUE
        # constraint. These lookups are indexed and cost well under a second.
        pairs = [
            pair for pair in selected if not database.exists_two_view_geometry(*pair)
        ]
        if len(pairs) != len(selected):
            logging.info(f"{len(selected) - len(pairs)} pairs already matched, "
                         f"matching the remaining {len(pairs)}")

        cameras = {camera.camera_id: camera for camera in database.read_all_cameras()}
        images = {image.image_id: image for image in database.read_all_images()}

        @lru_cache(maxsize=_FEATURE_CACHE_SIZE)
        def features_of(image_id):
            with database_lock:
                keypoints = database.read_keypoints(image_id)
                descriptors = database.read_descriptors(image_id).to_float().data
            return LocalFeatures(
                keypoints=keypoints[:, :2].astype(np.float64),
                descriptors=np.asarray(descriptors),
            )

        def match(pair):
            image_id1, image_id2 = pair
            features1 = features_of(image_id1)
            features2 = features_of(image_id2)
            camera1 = cameras[images[image_id1].camera_id]
            camera2 = cameras[images[image_id2].camera_id]
            matches = matcher.match(
                features1,
                (camera1.width, camera1.height),
                features2,
                (camera2.width, camera2.height),
            )
            if len(matches) == 0:
                return None
            two_view_geometry = pycolmap.estimate_two_view_geometry(
                camera1,
                features1.keypoints,
                camera2,
                features2.keypoints,
                matches,
                two_view_geometry_options,
            )
            return matches, two_view_geometry

        for pair, matched in zip(
            pairs,
            map_with_progress(match, pairs, "Matched", num_threads, _LOG_EVERY),
            strict=True,
        ):
            if matched is None:
                continue
            matches, two_view_geometry = matched
            with database_lock:
                database.write_matches(*pair, matches)
                database.write_two_view_geometry(*pair, two_view_geometry)
            num_pairs_written += 1

    logging.info(f"Wrote matches for {num_pairs_written} of {len(pairs)} pairs")
    return num_pairs_written
