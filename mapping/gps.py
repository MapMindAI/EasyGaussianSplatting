"""The capture's GPS track, as database pose priors and a map-to-world transform.

Insta360 writes a roughly 1 Hz WGS84 track into the capture's telemetry, which
exiftool reads out of either the `.insv` or its much smaller `.lrv` proxy.
Projecting the track into the capture's UTM zone puts it in the same metric,
gravity-aligned frame the reconstruction is levelled into, so aligning the two
is a similarity transform with the up axis already agreed on.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pycolmap
from pycolmap import logging

from .panorama_database import parse_image_name

# Horizontal fixes are good to a few metres; the altitude on the same track
# wanders over ten.
HORIZONTAL_STANDARD_DEVIATION = 3.0
VERTICAL_STANDARD_DEVIATION = 10.0

# A fix this far from a sampled frame is not evidence about where it was.
MAX_INTERPOLATION_GAP = 2.0

# The floor keeps a tight fit from discarding frames over centimetres of noise.
OUTLIER_RESIDUAL_FACTOR = 3.0
OUTLIER_RESIDUAL_FLOOR = 2.0

TRANSFORM_FILENAME = "local_to_world.json"


@dataclass(frozen=True)
class GpsFix:
    """One fix, timed from the start of the capture."""

    seconds: float
    latitude: float
    longitude: float
    altitude: float


# WGS84, and the UTM grid defined on it.
EARTH_RADIUS = 6378137.0
FLATTENING = 1.0 / 298.257223563
SCALE_FACTOR = 0.9996
FALSE_EASTING = 500000.0
SOUTHERN_FALSE_NORTHING = 10000000.0


def utm_epsg(latitude, longitude):
    """The EPSG code of the WGS84 UTM zone holding this coordinate."""
    zone = int(math.floor((longitude + 180.0) / 6.0)) + 1
    return (32600 if latitude >= 0.0 else 32700) + zone


def crs_name(epsg):
    return f"WGS 84 / UTM zone {epsg % 100}{'N' if epsg < 32700 else 'S'}"


def project_to_utm(latitudes, longitudes, epsg):
    """Eastings and northings of WGS84 degrees, in the zone `epsg` names.

    Snyder's transverse Mercator series, which holds to a millimetre inside a
    zone -- three orders finer than the fixes going through it, and small
    enough not to be worth a projection library in the image.
    """
    eccentricity = 2.0 * FLATTENING - FLATTENING**2
    second_eccentricity = eccentricity / (1.0 - eccentricity)
    central_meridian = math.radians((epsg % 100 - 1) * 6 - 180 + 3)

    latitude = np.radians(np.asarray(latitudes, dtype=float))
    longitude = np.radians(np.asarray(longitudes, dtype=float))
    sine, cosine, tangent = np.sin(latitude), np.cos(latitude), np.tan(latitude)

    normal = EARTH_RADIUS / np.sqrt(1.0 - eccentricity * sine**2)
    squared_tangent = tangent**2
    curvature = second_eccentricity * cosine**2
    offset = (longitude - central_meridian) * cosine

    meridian_arc = EARTH_RADIUS * (
        (1.0 - eccentricity / 4.0 - 3.0 * eccentricity**2 / 64.0
         - 5.0 * eccentricity**3 / 256.0) * latitude
        - (3.0 * eccentricity / 8.0 + 3.0 * eccentricity**2 / 32.0
           + 45.0 * eccentricity**3 / 1024.0) * np.sin(2.0 * latitude)
        + (15.0 * eccentricity**2 / 256.0
           + 45.0 * eccentricity**3 / 1024.0) * np.sin(4.0 * latitude)
        - (35.0 * eccentricity**3 / 3072.0) * np.sin(6.0 * latitude)
    )

    easting = FALSE_EASTING + SCALE_FACTOR * normal * (
        offset
        + (1.0 - squared_tangent + curvature) * offset**3 / 6.0
        + (5.0 - 18.0 * squared_tangent + squared_tangent**2 + 72.0 * curvature
           - 58.0 * second_eccentricity) * offset**5 / 120.0
    )
    northing = SCALE_FACTOR * (
        meridian_arc + normal * tangent * (
            offset**2 / 2.0
            + (5.0 - squared_tangent + 9.0 * curvature
               + 4.0 * curvature**2) * offset**4 / 24.0
            + (61.0 - 58.0 * squared_tangent + squared_tangent**2
               + 600.0 * curvature - 330.0 * second_eccentricity) * offset**6 / 720.0
        )
    )
    if epsg >= 32700:
        northing = northing + SOUTHERN_FALSE_NORTHING
    return easting, northing


def _exiftool(*arguments):
    result = subprocess.run(
        ["exiftool", "-api", "largefilesupport=1", *arguments],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0 and not result.stdout:
        raise RuntimeError(f"exiftool failed: {result.stderr.strip()}")
    return result.stdout


def _parse_time(text):
    text = text.strip().removesuffix("Z")
    layout = "%Y:%m:%d %H:%M:%S.%f" if "." in text else "%Y:%m:%d %H:%M:%S"
    return datetime.strptime(text, layout).replace(tzinfo=timezone.utc)


def capture_start(video_path):
    """The capture's start, which its GPS timestamps are absolute against.

    QuickTime creation dates carry no zone and Insta360 writes them in UTC,
    which is the same clock as the GPS timestamps.
    """
    text = _exiftool("-s3", "-CreateDate", str(video_path)).strip()
    if not text:
        raise RuntimeError(f"{video_path} has no CreateDate to time its GPS against")
    return _parse_time(text.splitlines()[0])


def parse_fixes(output, start):
    """Fixes from exiftool's `-p` rows, deduplicated and ordered by time.

    The track repeats each fix across the frames it covers, so the same
    timestamp arrives many times over.
    """
    by_time = {}
    for line in output.splitlines():
        fields = line.split(",")
        if len(fields) != 4 or not fields[1].strip():
            continue
        try:
            moment = _parse_time(fields[0])
            latitude, longitude, altitude = (float(value) for value in fields[1:])
        except ValueError:
            continue
        by_time[moment] = (latitude, longitude, altitude)
    return [
        GpsFix((moment - start).total_seconds(), *by_time[moment])
        for moment in sorted(by_time)
    ]


def read_fixes(video_path):
    """The capture's GPS track, timed from its start."""
    output = _exiftool(
        "-ee", "-n",
        "-p", "$GPSDateTime,$GPSLatitude,$GPSLongitude,$GPSAltitude",
        str(video_path),
    )
    fixes = parse_fixes(output, capture_start(video_path))
    # Fatal, unlike a track too sparse to fit: --gps-video is opt-in, so a
    # capture with no track at all is the wrong file rather than a bad fix.
    if not fixes:
        raise RuntimeError(f"{video_path} carries no GPS track")
    return fixes


class GpsTrack:
    """A capture's GPS track in UTM metres, sampled by capture time."""

    def __init__(self, fixes):
        fixes = sorted(fixes, key=lambda fix: fix.seconds)
        self.epsg = utm_epsg(fixes[0].latitude, fixes[0].longitude)
        eastings, northings = project_to_utm(
            [fix.latitude for fix in fixes],
            [fix.longitude for fix in fixes],
            self.epsg,
        )
        self.seconds = np.array([fix.seconds for fix in fixes])
        self.positions = np.column_stack(
            [eastings, northings, [fix.altitude for fix in fixes]]
        )

    def __len__(self):
        return len(self.seconds)

    @property
    def duration(self):
        return float(self.seconds[-1] - self.seconds[0])

    def position_at(self, seconds):
        """The interpolated position at `seconds`, or None where the track has
        nothing close enough to say.

        Off either end the nearest fix is the only neighbour, so the gap below
        rejects a time outside the track without a separate bounds check.
        """
        index = int(np.searchsorted(self.seconds, seconds))
        neighbours = [i for i in (index - 1, index) if 0 <= i < len(self.seconds)]
        if min(abs(self.seconds[i] - seconds) for i in neighbours) > MAX_INTERPOLATION_GAP:
            return None
        return np.array([
            np.interp(seconds, self.seconds, self.positions[:, axis])
            for axis in range(3)
        ])


def read_track(video_path):
    track = GpsTrack(read_fixes(video_path))
    logging.info(
        f"Read {len(track)} GPS fixes over {track.duration:.1f} s from "
        f"{video_path}, projected to EPSG:{track.epsg}"
    )
    return track


def position_of(track, image_name, seconds_per_frame):
    """Where the track puts a database image, or None if it cannot place it."""
    _, index = parse_image_name(image_name)
    return track.position_at(index * seconds_per_frame)


def write_pose_priors(database_path, track, seconds_per_frame):
    """Writes one UTM position prior per image, keeping any already there.

    Every face of a panorama is rendered about the same optical centre, so all
    of them take the frame's position.

    These priors are recorded, not solved with: `pycolmap.global_mapping` reads
    pose priors only for their gravity, and the position-prior bundle adjuster
    belongs to the incremental mapper. Georeferencing is `align_to_track`,
    which fits the solved map afterwards.
    """
    covariance = np.diag(np.square([
        HORIZONTAL_STANDARD_DEVIATION,
        HORIZONTAL_STANDARD_DEVIATION,
        VERTICAL_STANDARD_DEVIATION,
    ]))
    written = 0
    with pycolmap.Database.open(database_path) as database:
        if database.num_pose_priors():
            logging.info(f"{database_path} already holds pose priors, keeping them")
            return 0
        # Otherwise sqlite commits each of the several thousand rows on its own.
        with pycolmap.DatabaseTransaction(database):
            for image in database.read_all_images():
                position = position_of(track, image.name, seconds_per_frame)
                if position is None:
                    continue
                database.write_pose_prior(pycolmap.PosePrior(
                    corr_data_id=image.data_id,
                    position=position,
                    position_covariance=covariance,
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                ))
                written += 1
    logging.info(f"Wrote {written} GPS pose priors to {database_path}")
    return written


def horizontal_similarity(source, target):
    """The scale, yaw and translation best taking `source` onto `target`.

    Restricted to a turn about the vertical so that it cannot undo the gravity
    levelling, and scaled off the horizontal because the altitudes are the
    noisiest part of the track.
    """
    source_centre, target_centre = source.mean(axis=0), target.mean(axis=0)
    centred_source = source - source_centre
    centred_target = target - target_centre

    dot = float(np.sum(centred_source[:, :2] * centred_target[:, :2]))
    cross = float(np.sum(
        centred_source[:, 0] * centred_target[:, 1]
        - centred_source[:, 1] * centred_target[:, 0]
    ))
    spread = float(np.sum(centred_source[:, :2] ** 2))
    if spread <= 0.0 or (dot == 0.0 and cross == 0.0):
        return None

    yaw = math.atan2(cross, dot)
    scale = math.hypot(dot, cross) / spread
    rotation = pycolmap.Rotation3d(
        np.array([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)])
    )
    return pycolmap.Sim3d(scale, rotation, target_centre - scale * (rotation * source_centre))


def _residuals(transform, source, target):
    moved = np.array([transform * point for point in source])
    return np.linalg.norm((moved - target)[:, :2], axis=1)


def fit_similarity(source, target):
    """`horizontal_similarity` refit without the frames it fits worst."""
    transform = horizontal_similarity(source, target)
    if transform is None:
        return None, None
    residuals = _residuals(transform, source, target)
    limit = max(OUTLIER_RESIDUAL_FACTOR * float(np.median(residuals)),
                OUTLIER_RESIDUAL_FLOOR)
    kept = residuals <= limit
    # Refitting on a handful of frames is worse than keeping every one of them.
    if kept.sum() >= 3 and not kept.all():
        refit = horizontal_similarity(source[kept], target[kept])
        if refit is not None:
            transform, residuals = refit, _residuals(refit, source, target)
    return transform, residuals


def reference_images(reconstruction, reference_face):
    """The registered reference-face images, one per placed panorama.

    Every face of a panorama shares its optical centre, so the reference face
    alone carries each frame's position exactly once.
    """
    return [
        image for image in reconstruction.images.values()
        if parse_image_name(image.name)[0] == reference_face and image.has_pose
    ]


def paired_positions(images, track, seconds_per_frame):
    """Frame centres paired with where the track puts them, for those it can."""
    map_positions, world_positions = [], []
    for image in images:
        position = position_of(track, image.name, seconds_per_frame)
        if position is None:
            continue
        map_positions.append(image.projection_center())
        world_positions.append(position)
    return np.array(map_positions), np.array(world_positions)


def align_to_track(reconstruction, track, reference_face, seconds_per_frame):
    """Puts `reconstruction` in UTM metres about its own centre, in place.

    Returns the Sim3d taking the result back to UTM, which is what
    `save_transform` records, or None when the track cannot place the map.

    The centre is the mean of every registered frame, not only the frames the
    track could place: a capture whose GPS covers part of the run would
    otherwise be centred on that part rather than on the map.
    """
    images = reference_images(reconstruction, reference_face)
    map_positions, world_positions = paired_positions(
        images, track, seconds_per_frame
    )
    if len(map_positions) < 3:
        logging.warning(
            f"Only {len(map_positions)} frames have both a pose and a GPS fix, "
            "leaving the map unaligned"
        )
        return None

    transform, residuals = fit_similarity(map_positions, world_positions)
    if transform is None:
        logging.warning("The GPS track has no extent to align to, leaving the map unaligned")
        return None

    reconstruction.transform(transform)
    centre = np.array([image.projection_center() for image in images]).mean(axis=0)
    reconstruction.transform(pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), -centre))

    logging.info(
        f"Aligned {len(map_positions)} frames to GPS: scale {transform.scale:.4f}, "
        f"horizontal residual {float(np.sqrt(np.mean(residuals ** 2))):.2f} m RMS "
        f"(max {float(residuals.max()):.2f} m)"
    )
    return pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), centre), residuals


def save_transform(workspace_path, local_to_world, epsg, residuals, source):
    """Writes the local-to-world transform beside the reconstruction."""
    path = Path(workspace_path) / TRANSFORM_FILENAME
    path.write_text(json.dumps({
        "epsg": epsg,
        "crs": crs_name(epsg),
        "scale": float(local_to_world.scale),
        "rotation_quaternion_xyzw": local_to_world.rotation.quat.tolist(),
        "translation": local_to_world.translation.tolist(),
        "frames_aligned": len(residuals),
        "horizontal_residual_rms_metres": float(np.sqrt(np.mean(residuals ** 2))),
        "gps_source": str(source),
    }, indent=2) + "\n")
    logging.info(f"Wrote {path}")
    return path
