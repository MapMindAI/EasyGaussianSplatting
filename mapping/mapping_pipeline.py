#!/usr/bin/env python3
"""Reconstruct a panorama video with learned features and global SfM.

Four stages into one workspace: a cube-map rig database, SuperPoint/SALAD
extraction, LightGlue matching, and `pycolmap.global_mapping`. The first three
pick up where a previous run stopped rather than redoing their work; the
mapping always reruns. The result is a cube-map reconstruction gsplat can train
on directly.

See doc/panorama_mapping.md for the pipeline, the models it needs served, and
the pair-selection knobs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycolmap
from pycolmap import logging

from .triton.extraction import GLOBAL_FEATURES_FILENAME, extract_features
from .triton.matching import PairSelection, match_features
from .triton.clients import (
    FeatureMatcher,
    GlobalFeatureExtractor,
    LocalFeatureExtractor,
)
from . import gps
from .panorama_database import (
    DEFAULT_FACES,
    FACE_AXES,
    REFERENCE_FACE,
    RIG_DOWN,
    build_database,
    parse_faces,
)


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


_WORLD_DOWN = np.array([0.0, 0.0, -1.0])
# Half a turn about X, for a map solved upside down: the shortest arc has no
# determined axis there, and every horizontal one serves equally.
_HALF_TURN = pycolmap.Rotation3d(np.array([1.0, 0.0, 0.0, 0.0]))


def _rotation_to_world_down(down):
    """The shortest-arc rotation taking the unit vector `down` onto -Z."""
    cosine = float(np.dot(down, _WORLD_DOWN))
    if cosine < -1.0 + 1e-9:
        return _HALF_TURN
    quaternion = np.array([*np.cross(down, _WORLD_DOWN), 1.0 + cosine])
    return pycolmap.Rotation3d(quaternion / np.linalg.norm(quaternion))


def align_up_axis(reconstruction):
    """Rotates `reconstruction` in place so that +Z points up.

    Global SfM fixes the world frame on whichever image it starts from, leaving
    the map tilted arbitrarily. Every panorama here is captured upright, so the
    rig's own down axis is gravity: averaging it over the registered frames
    gives gravity in world axes, and one rotation takes that to -Z. Over a real
    capture that axis holds to half a degree per frame, which is what makes the
    average meaningful. Positions and scale are left as the mapper solved them.
    """
    down = np.zeros(3)
    for frame in reconstruction.frames.values():
        if frame.has_pose:
            down += frame.rig_from_world.rotation.inverse() * RIG_DOWN
    if np.linalg.norm(down) < 1e-9:
        logging.warning("No registered frames to read gravity from, leaving the map as solved")
        return
    down /= np.linalg.norm(down)
    reconstruction.transform(
        pycolmap.Sim3d(1.0, _rotation_to_world_down(down), np.zeros(3))
    )
    logging.info(f"Rotated {np.round(down, 3)} to -Z, so the map is +Z up")


def run_global_mapping(workspace_path, track=None, seconds_per_frame=None,
                       gps_source=None):
    """Reconstructs `<workspace_path>/database.db` into `sparse/`.

    Levelling runs before any GPS alignment, which is then restricted to a turn
    about the vertical so it keeps that levelling: the track's altitudes are far
    noisier than the gravity the rig itself reports.
    """
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
        align_up_axis(model)
        if track is not None:
            aligned = gps.align_to_track(
                model, track, REFERENCE_FACE, seconds_per_frame
            )
            # Only the first model shares the workspace's transform file, and it
            # is the one the later stages read out of sparse/0.
            if aligned is not None and index == 0:
                local_to_world, residuals = aligned
                gps.save_transform(
                    workspace_path, local_to_world, track.epsg, residuals, gps_source
                )
        # Rewritten over what global_mapping just wrote, which is the untilted map.
        model.write(sparse_dir / str(index))
        logging.info(
            f"Wrote {sparse_dir / str(index)}: {model.num_reg_images()} images, "
            f"{model.num_points3D()} points"
        )


def discover_videos(workspace_path):
    """MP4 videos under the mapping workspace, in reproducible path order."""
    return sorted(
        path for path in Path(workspace_path).rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def capture_key(path):
    """The timestamp and sequence shared by a VID/LRV capture pair."""
    parts = Path(path).stem.split("_")
    if len(parts) != 5 or parts[0] not in {"VID", "LRV"}:
        return None
    return parts[1], parts[2], parts[4]


def find_lrv(video_path, search_path=None):
    """The matching LRV under `search_path`, or beside `video_path` by default."""
    key = capture_key(video_path)
    if key is None:
        return None
    search_path = Path(search_path) if search_path is not None else video_path.parent
    matches = [
        path for path in search_path.rglob("*")
        if path.is_file() and path.suffix.lower() == ".lrv" and capture_key(path) == key
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple LRV files match {video_path}: {matches}")
    return matches[0] if matches else None


def run_pipeline(workspace_path, triton_url, selection, frame_rate=2.0,
                 face_size=None, faces=DEFAULT_FACES, keypoint_threshold=0.015,
                 match_threshold=0.2, num_threads=8):
    """Reconstructs all MP4 videos under `workspace_path` into `sparse/0/`."""
    workspace_path = Path(workspace_path)
    video_paths = discover_videos(workspace_path)
    if not video_paths:
        raise RuntimeError(f"No MP4 videos found under {workspace_path}")
    videos = build_database(video_paths, workspace_path, frame_rate, face_size, faces)

    tracks = []
    for video in videos:
        lrv_path = find_lrv(video.path, workspace_path)
        if lrv_path is None:
            logging.warning(f"No matching LRV for {video.path}; skipping GPS alignment")
            continue
        tracks.append((
            video.start_frame_index,
            video.stop_frame_index,
            gps.read_track(lrv_path),
            video.seconds_per_frame,
            lrv_path,
        ))
    track = gps.CaptureTracks(tracks) if tracks else None
    if track is not None:
        gps.write_pose_priors(workspace_path / "database.db", track, None)

    # Skipped in the caller rather than in extract_features, so a finished
    # workspace never opens a connection to the inference server.
    if (workspace_path / GLOBAL_FEATURES_FILENAME).exists():
        logging.info("Global-descriptor sidecar already exists, skipping extraction")
    else:
        extract_features(
            workspace_path,
            LocalFeatureExtractor(triton_url, keypoint_threshold),
            GlobalFeatureExtractor(triton_url),
            num_threads,
        )

    match_features(
        workspace_path,
        FeatureMatcher(triton_url, match_threshold),
        selection,
        num_threads,
    )

    # Always resolved, unlike the stages above: it is the cheap one, and it is
    # where levelling and GPS alignment happen, so a rerun has to redo it for a
    # changed track or a changed alignment to reach the saved model.
    source = ", ".join(map(str, track.sources)) if track is not None else None
    run_global_mapping(workspace_path, track, None, source)
    return workspace_path / "sparse" / "0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace_path", required=True,
                        help="directory containing MP4 videos and reconstruction output")
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
