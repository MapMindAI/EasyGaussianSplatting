#!/usr/bin/env python3
"""Reconstruct a panorama video with learned features and global SfM.

Four stages into one workspace: a cube-map rig database, SuperPoint/SALAD
extraction, LightGlue matching, and `pycolmap.global_mapping`. Each picks up
where a previous run stopped rather than redoing its work. The result is a
cube-map reconstruction gsplat can train on directly.

See doc/panorama_mapping.md for the pipeline, the models it needs served, and
the pair-selection knobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pycolmap
from pycolmap import logging

from .features.extraction import (
    GLOBAL_FEATURES_FILENAME,
    extract_features,
    largest_camera_size,
)
from .features.matching import PairSelection, match_features
from .features.triton_models import (
    FeatureMatcher,
    GlobalFeatureExtractor,
    LocalFeatureExtractor,
)
from .panorama_database import DEFAULT_FACES, FACE_AXES, build_database, parse_faces


# Weight each relative-rotation edge by its inlier match count instead of
# treating every image pair equally, so well-matched pairs dominate poorly
# matched ones in rotation averaging.
_ROTATION_AVERAGING_REWEIGHTING = (
    pycolmap.RotationAveragingReweighting.INLIER_MATCH_COUNT
)


def build_global_mapper_options():
    """Global-SfM options for a cube-map rig.

    The cube faces' intrinsics and their mounting on the rig are exact by
    construction -- they come from the reprojection, not from a calibration --
    so bundle adjustment holds both fixed. Two-view tracks are kept because
    retrieval pairs routinely see a point from exactly two images, and
    discarding those (the default) throws that coverage away.
    """
    options = pycolmap.GlobalPipelineOptions()
    options.mapper.refine_sensor_from_rig = False
    options.mapper.retriangulation.ignore_two_view_tracks = False
    options.mapper.rotation_averaging.reweighting = _ROTATION_AVERAGING_REWEIGHTING
    bundle_adjustment = options.mapper.bundle_adjustment
    bundle_adjustment.refine_focal_length = False
    bundle_adjustment.refine_principal_point = False
    bundle_adjustment.refine_extra_params = False
    bundle_adjustment.print_summary = True
    return options


def run_global_mapping(workspace_path):
    """Reconstructs `<workspace_path>/database.db` into `sparse/`."""
    workspace_path = Path(workspace_path)
    sparse_dir = workspace_path / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    reconstructions = pycolmap.global_mapping(
        workspace_path / "database.db",
        workspace_path / "images",
        sparse_dir,
        build_global_mapper_options(),
    )
    if not reconstructions:
        raise RuntimeError("Global mapping produced no reconstruction")
    for index, model in sorted(reconstructions.items()):
        logging.info(
            f"Wrote {sparse_dir / str(index)}: {model.num_reg_images()} images, "
            f"{model.num_points3D()} points"
        )


def run_pipeline(video_path, workspace_path, triton_url, selection, frame_rate=2.0,
                 face_size=None, faces=DEFAULT_FACES, keypoint_threshold=0.015,
                 match_threshold=0.2, num_threads=8):
    """Reconstructs `video_path` into `workspace_path`, and returns its
    `sparse/0/` directory."""
    workspace_path = Path(workspace_path)
    build_database(video_path, workspace_path, frame_rate, face_size, faces)

    # Skipped in the caller rather than in extract_features, so a finished
    # workspace never opens a connection to the inference server.
    if (workspace_path / GLOBAL_FEATURES_FILENAME).exists():
        logging.info("Global-descriptor sidecar already exists, skipping extraction")
    else:
        with pycolmap.Database.open(workspace_path / "database.db") as database:
            max_image_size = largest_camera_size(database)
        extract_features(
            workspace_path,
            LocalFeatureExtractor(triton_url, max_image_size, keypoint_threshold),
            GlobalFeatureExtractor(triton_url),
            num_threads,
        )

    match_features(
        workspace_path,
        FeatureMatcher(triton_url, match_threshold),
        selection,
        num_threads,
    )

    model_dir = workspace_path / "sparse" / "0"
    if (model_dir / "cameras.bin").exists():
        logging.info(f"{model_dir} already exists, skipping global mapping")
    else:
        run_global_mapping(workspace_path)
    return model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True,
                        help="stitched equirectangular video")
    parser.add_argument("--workspace_path", required=True,
                        help="directory to reconstruct into")
    parser.add_argument("--triton-url", default="127.0.0.1:8011",
                        help="Triton gRPC endpoint (default: 127.0.0.1:8011)")
    parser.add_argument("--frame-rate", type=float, default=2.0,
                        help="frames per second to sample from the video (default: 2.0)")
    parser.add_argument("--face-size", type=int, default=None,
                        help="width/height in pixels of each cube face "
                             "(default: a quarter of the video width, which keeps "
                             "the panorama's angular resolution)")
    parser.add_argument("--faces", default=",".join(DEFAULT_FACES),
                        help=f"comma-separated subset of {','.join(FACE_AXES)} "
                             f"(default: {','.join(DEFAULT_FACES)})")
    parser.add_argument("--keypoint-threshold", type=float, default=0.015,
                        help="SuperPoint detection score threshold (default: 0.015)")
    parser.add_argument("--match-threshold", type=float, default=0.2,
                        help="LightGlue match score threshold (default: 0.2)")
    parser.add_argument("--num-sequential", type=int, default=PairSelection.num_sequential,
                        help="same-face images ahead of each image to match "
                             f"(default: {PairSelection.num_sequential})")
    parser.add_argument("--num-retrieval", type=int, default=PairSelection.num_retrieval,
                        help="nearest images by SALAD descriptor to match "
                             f"(default: {PairSelection.num_retrieval})")
    parser.add_argument("--num-retrieval-excluded", type=int,
                        default=PairSelection.num_retrieval_excluded,
                        help="same-face frames on either side that retrieval skips "
                             f"(default: {PairSelection.num_retrieval_excluded})")
    parser.add_argument("--num-threads", type=int, default=8,
                        help="worker threads calling the models (default: 8)")
    args = parser.parse_args()
    logging.info(f"Parameters: {vars(args)}")

    model_dir = run_pipeline(
        args.video_path,
        args.workspace_path,
        args.triton_url,
        PairSelection(
            num_sequential=args.num_sequential,
            num_retrieval=args.num_retrieval,
            num_retrieval_excluded=args.num_retrieval_excluded,
        ),
        frame_rate=args.frame_rate,
        face_size=args.face_size,
        faces=parse_faces(args.faces),
        keypoint_threshold=args.keypoint_threshold,
        match_threshold=args.match_threshold,
        num_threads=args.num_threads,
    )
    logging.info(f"Reconstruction written to {model_dir}")


if __name__ == "__main__":
    main()
