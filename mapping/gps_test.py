"""Tests for the capture's GPS track and the alignment it drives."""

import json
import math
from datetime import datetime, timezone

import numpy as np
import pycolmap
import pytest

from mapping import gps

CAPTURE_START = datetime(2026, 9, 4, 7, 58, 49, tzinfo=timezone.utc)

# Rows as exiftool -ee -n -p prints them, including the repeats a real track
# carries and a row whose fix the INSV parser could not read.
EXIFTOOL_ROWS = """\
2026:09:04 07:58:49.528Z,22.7811764897309,113.512259637094,10.3035766085733
2026:09:04 07:58:49.528Z,22.7811764897309,113.512259637094,10.3035766085733
2026:09:04 07:58:51Z,22.7812764897309,113.512359637094,10.5
2026:09:04 07:58:50Z,22.7812264897309,113.512309637094,10.4
2026:09:04 07:58:52Z,,,
"""


def track_from(*offsets_and_positions):
    """A track placed by hand, in metres east/north of a reference fix."""
    fixes = []
    for seconds, east, north, up in offsets_and_positions:
        # ~1e-5 degrees is about a metre here, close enough to place fixes.
        fixes.append(gps.GpsFix(
            seconds=seconds,
            latitude=22.78 + north * 9.0e-6,
            longitude=113.51 + east * 9.75e-6,
            altitude=up,
        ))
    return gps.GpsTrack(fixes)


class Image:
    def __init__(self, name, centre, has_pose=True):
        self.name = name
        self.has_pose = has_pose
        self.centre = np.asarray(centre, dtype=float)

    def projection_center(self):
        return self.centre


class Reconstruction:
    """Only what the alignment reads, and a transform that moves the frames."""

    def __init__(self, *images):
        self.images = {index: image for index, image in enumerate(images)}
        self.applied = []

    def transform(self, sim3d):
        self.applied.append(sim3d)
        for image in self.images.values():
            image.centre = sim3d * image.centre


# --- UTM zones ------------------------------------------------------------


@pytest.mark.parametrize(
    "latitude, longitude, expected",
    [
        (22.78, 113.51, 32649),   # the example capture, zone 49 north
        (51.5, -0.12, 32630),     # London, zone 30 north
        (-33.87, 151.2, 32756),   # Sydney, zone 56 south
        (0.0, 0.0, 32631),        # the zone boundary at the prime meridian
        (10.0, -180.0, 32601),    # the first zone
        (10.0, 179.9, 32660),     # the last zone
    ],
)
def test_utm_epsg_picks_the_zone(latitude, longitude, expected):
    assert gps.utm_epsg(latitude, longitude) == expected


@pytest.mark.parametrize(
    "epsg, expected",
    [(32649, "WGS 84 / UTM zone 49N"), (32630, "WGS 84 / UTM zone 30N"),
     (32756, "WGS 84 / UTM zone 56S"), (32601, "WGS 84 / UTM zone 1N")],
)
def test_crs_name_reads_as_the_epsg_registry_does(epsg, expected):
    assert gps.crs_name(epsg) == expected


# Eastings and northings PROJ gives for these coordinates. The projection here
# has to agree with the wider world's, so the numbers come from outside it.
PROJ_REFERENCE = [
    (22.7811764897309, 113.512259637094, 32649, 757931.6067, 2521486.9640),
    (22.7807835010722, 113.513714031335, 32649, 758081.7357, 2521445.9706),
    (51.5, -0.12, 32630, 699889.8070, 5709362.2928),
    (-33.87, 151.2, 32756, 333510.6501, 6250800.2412),
    (0.0, 0.0, 32631, 166021.4431, 0.0000),
    (71.0, 25.0, 32635, 427338.6673, 7878596.0016),
    (-54.8, -68.3, 32719, 545000.0534, 3927239.3813),
    (10.0, -177.0, 32601, 500000.0000, 1105412.4913),
]


@pytest.mark.parametrize("latitude, longitude, epsg, easting, northing", PROJ_REFERENCE)
def test_the_projection_agrees_with_proj(latitude, longitude, epsg, easting, northing):
    east, north = gps.project_to_utm([latitude], [longitude], epsg)

    assert float(east[0]) == pytest.approx(easting, abs=0.001)
    assert float(north[0]) == pytest.approx(northing, abs=0.001)


def test_the_projection_takes_whole_tracks_at_once():
    latitudes = [case[0] for case in PROJ_REFERENCE if case[2] == 32649]
    longitudes = [case[1] for case in PROJ_REFERENCE if case[2] == 32649]

    east, north = gps.project_to_utm(latitudes, longitudes, 32649)

    assert east.shape == (2,) and north.shape == (2,)
    assert float(east[0]) == pytest.approx(757931.6067, abs=0.001)


def test_a_metre_on_the_ground_is_a_metre_on_the_grid():
    """The grid is metric, which is what makes the fitted scale metres.

    Measured as a distance: away from the central meridian a step due north
    also moves the easting, since the grid's north is not the meridian's.
    """
    degree = 1.0 / 111320.0
    east, north = gps.project_to_utm([22.78, 22.78 + degree], [113.51, 113.51], 32649)

    step = math.hypot(float(east[1] - east[0]), float(north[1] - north[0]))
    assert step == pytest.approx(1.0, abs=0.01)


# --- parsing exiftool's rows ---------------------------------------------


def test_fixes_are_timed_from_the_capture_start():
    fixes = gps.parse_fixes(EXIFTOOL_ROWS, CAPTURE_START)

    assert [round(fix.seconds, 3) for fix in fixes] == [0.528, 1.0, 2.0]


def test_repeated_and_unreadable_rows_are_dropped():
    fixes = gps.parse_fixes(EXIFTOOL_ROWS, CAPTURE_START)

    # Five rows in: one repeat, one out of order, one with no fix.
    assert len(fixes) == 3
    assert fixes[0].latitude == pytest.approx(22.7811764897309)
    assert fixes[0].altitude == pytest.approx(10.3035766085733)


def test_rows_without_a_track_parse_to_nothing():
    assert gps.parse_fixes("", CAPTURE_START) == []
    assert gps.parse_fixes("Warning: no GPS\n", CAPTURE_START) == []


# --- the projected track --------------------------------------------------


def test_the_track_projects_into_the_captures_utm_zone():
    track = track_from((0.0, 0.0, 0.0, 5.0), (1.0, 10.0, 0.0, 5.0))

    assert track.epsg == 32649
    assert len(track) == 2
    # A metre of longitude is a metre of easting, to within the approximation.
    assert track.positions[1, 0] - track.positions[0, 0] == pytest.approx(10.0, abs=0.5)


def test_the_track_interpolates_between_fixes():
    track = track_from((0.0, 0.0, 0.0, 10.0), (2.0, 20.0, 0.0, 20.0))

    midpoint = track.position_at(1.0)

    assert midpoint[0] == pytest.approx(track.positions[:, 0].mean(), abs=1e-6)
    assert midpoint[2] == pytest.approx(15.0)


def test_the_track_says_nothing_far_outside_itself():
    track = track_from((10.0, 0.0, 0.0, 5.0), (12.0, 5.0, 0.0, 5.0))

    assert track.position_at(-5.0) is None
    assert track.position_at(100.0) is None
    # Just past either end is still the capture, within the sampling gap.
    assert track.position_at(9.0) is not None
    assert track.position_at(13.0) is not None


def test_the_track_says_nothing_inside_a_long_gap():
    track = track_from((0.0, 0.0, 0.0, 5.0), (60.0, 100.0, 0.0, 5.0))

    assert track.position_at(30.0) is None
    assert track.position_at(1.0) is not None


def test_duration_spans_the_fixes():
    track = track_from((0.5, 0.0, 0.0, 5.0), (8.5, 10.0, 0.0, 5.0))

    assert track.duration == pytest.approx(8.0)


# --- the horizontal fit ---------------------------------------------------


def yaw_rotation(degrees):
    half = math.radians(degrees) / 2.0
    return pycolmap.Rotation3d(np.array([0.0, 0.0, math.sin(half), math.cos(half)]))


SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.5]])


@pytest.mark.parametrize("scale", [1.0, 2.5, 0.4])
@pytest.mark.parametrize("degrees", [0.0, 30.0, -95.0, 179.0])
def test_the_fit_recovers_a_similarity_it_is_given(scale, degrees):
    truth = pycolmap.Sim3d(scale, yaw_rotation(degrees), np.array([100.0, -50.0, 7.0]))
    target = np.array([truth * point for point in SQUARE])

    fitted = gps.horizontal_similarity(SQUARE, target)

    assert fitted.scale == pytest.approx(scale, rel=1e-6)
    assert np.array([fitted * point for point in SQUARE]) == pytest.approx(target, abs=1e-6)


def test_the_fit_only_turns_about_the_vertical():
    """A tilt in the target must not tilt the map, which gravity already fixed."""
    target = SQUARE + np.array([0.0, 0.0, 1.0]) * SQUARE[:, [0]] * 5.0

    fitted = gps.horizontal_similarity(SQUARE, target)

    assert fitted.rotation.quat[:2] == pytest.approx([0.0, 0.0], abs=1e-9)
    assert (fitted.rotation * np.array([0.0, 0.0, 1.0])) == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-9
    )


def test_the_fit_takes_its_scale_from_the_horizontal_alone():
    """Altitude is the noisiest part of the track, so it must not set scale."""
    target = SQUARE * np.array([3.0, 3.0, 1.0])

    fitted = gps.horizontal_similarity(SQUARE, target)

    assert fitted.scale == pytest.approx(3.0, rel=1e-6)


def test_a_track_with_no_extent_cannot_be_fitted():
    stationary = np.zeros((4, 3))

    assert gps.horizontal_similarity(stationary, SQUARE) is None


def test_the_refit_drops_the_frame_that_fits_worst():
    source = np.array([[float(i), 0.0, 0.0] for i in range(12)])
    target = source.copy()
    target[6] += np.array([40.0, 25.0, 0.0])  # one bad fix

    with_outlier = gps.horizontal_similarity(source, target)
    refitted, residuals = gps.fit_similarity(source, target)

    assert refitted.scale == pytest.approx(1.0, rel=1e-6)
    assert residuals[6] > 40.0
    assert np.median(residuals) < np.median(
        gps._residuals(with_outlier, source, target)
    )


def test_a_clean_fit_keeps_every_frame():
    source = np.array([[float(i), 0.0, 0.0] for i in range(6)])
    truth = pycolmap.Sim3d(2.0, yaw_rotation(20.0), np.array([5.0, 6.0, 7.0]))
    target = np.array([truth * point for point in source])

    _, residuals = gps.fit_similarity(source, target)

    assert residuals == pytest.approx(np.zeros(6), abs=1e-6)


# --- aligning a reconstruction -------------------------------------------


def straight_run(count=8):
    """A capture driven due east at ten metres a second, sampled once a second."""
    spacing = 10.0
    track = track_from(*[(float(i), i * spacing, 0.0, 5.0) for i in range(count)])
    images = [
        Image(f"front/{i:06d}.jpg", [i * spacing / 2.0, 0.0, 0.0]) for i in range(count)
    ]
    return track, images


def test_alignment_scales_and_centres_the_map():
    track, images = straight_run()
    reconstruction = Reconstruction(*images)
    # Read before aligning, which moves the frames.
    map_span = np.linalg.norm(
        images[-1].projection_center() - images[0].projection_center()
    )

    local_to_world, residuals = gps.align_to_track(reconstruction, track, "front", 1.0)

    similarity, centring = reconstruction.applied
    # The map runs at half the track's spacing, so it doubles. The expected
    # factor comes off the projected track, whose metres per degree are only
    # approximated by the fixes above.
    track_span = np.linalg.norm((track.positions[-1] - track.positions[0])[:2])
    assert similarity.scale == pytest.approx(track_span / map_span, rel=1e-6)
    assert similarity.scale == pytest.approx(2.0, rel=1e-2)
    # Not exactly zero: equal steps of longitude are not exactly equal steps
    # of easting, so the placed fixes are microns off a straight line.
    assert residuals == pytest.approx(np.zeros(len(images)), abs=1e-3)
    # Centring cancels the aligned centre, and the saved transform restores it.
    assert centring.translation == pytest.approx(-local_to_world.translation)
    assert local_to_world.scale == pytest.approx(1.0)


def test_the_centred_map_is_about_the_origin():
    track, images = straight_run()
    reconstruction = Reconstruction(*images)

    gps.align_to_track(reconstruction, track, "front", 1.0)

    centres = [image.projection_center() for image in images]
    assert np.mean(centres, axis=0) == pytest.approx(np.zeros(3), abs=1e-6)


def test_the_centre_spans_the_frames_the_track_could_not_place():
    """A track covering part of the run still centres the whole map."""
    track, images = straight_run()
    # Sampled long after the track ends, so they take no part in the fit.
    unplaced = [Image(f"front/{i:06d}.jpg", [i * 5.0, 40.0, 0.0]) for i in (400, 401)]
    reconstruction = Reconstruction(*images, *unplaced)

    _, residuals = gps.align_to_track(reconstruction, track, "front", 1.0)

    assert len(residuals) == len(images)
    placed_only = np.mean([image.projection_center() for image in images], axis=0)
    everything = np.mean(
        [image.projection_center() for image in images + unplaced], axis=0
    )
    assert everything == pytest.approx(np.zeros(3), abs=1e-6)
    assert np.linalg.norm(placed_only) > 1.0


def test_only_the_reference_face_constrains_the_alignment():
    track, images = straight_run()
    extra = [Image(f"back/{i:06d}.jpg", [0.0, 0.0, 0.0]) for i in range(len(images))]
    reconstruction = Reconstruction(*images, *extra)

    _, residuals = gps.align_to_track(reconstruction, track, "front", 1.0)

    assert len(residuals) == len(images)


def test_frames_without_a_pose_are_left_out():
    track, images = straight_run()
    images[3].has_pose = False
    reconstruction = Reconstruction(*images)

    _, residuals = gps.align_to_track(reconstruction, track, "front", 1.0)

    assert len(residuals) == len(images) - 1


def test_too_few_placed_frames_leaves_the_map_alone():
    track, images = straight_run(count=2)
    reconstruction = Reconstruction(*images)

    assert gps.align_to_track(reconstruction, track, "front", 1.0) is None
    assert reconstruction.applied == []


def test_frames_the_track_cannot_place_are_left_out():
    track, images = straight_run()
    # A sample far past the end of the track has no fix to pair with.
    reconstruction = Reconstruction(*images, Image("front/000900.jpg", [900.0, 0.0, 0.0]))

    _, residuals = gps.align_to_track(reconstruction, track, "front", 1.0)

    assert len(residuals) == len(images)


# --- the saved transform --------------------------------------------------


def test_the_saved_transform_reads_back_as_the_map_origin(tmp_path):
    origin = np.array([757931.61, 2521486.96, 11.0])
    local_to_world = pycolmap.Sim3d(1.0, pycolmap.Rotation3d(), origin)

    path = gps.save_transform(
        tmp_path, local_to_world, 32649, np.array([0.4, 0.6]), "capture.lrv"
    )

    saved = json.loads(path.read_text())
    assert path.name == gps.TRANSFORM_FILENAME
    assert saved["epsg"] == 32649
    assert "49N" in saved["crs"]
    assert saved["translation"] == pytest.approx(origin.tolist())
    assert saved["scale"] == pytest.approx(1.0)
    assert saved["frames_aligned"] == 2
    assert saved["horizontal_residual_rms_metres"] == pytest.approx(0.5099, abs=1e-3)
    assert saved["gps_source"] == "capture.lrv"


# --- pose priors ----------------------------------------------------------


def test_position_of_times_an_image_by_its_frame_index():
    track = track_from((0.0, 0.0, 0.0, 5.0), (2.0, 20.0, 0.0, 5.0))

    # Sampled every half second, so frame 2 is the fix a second in.
    assert gps.position_of(track, "front/000002.jpg", 0.5) == pytest.approx(
        track.position_at(1.0)
    )
    assert gps.position_of(track, "down/000002.jpg", 0.5) == pytest.approx(
        track.position_at(1.0)
    )


def test_position_of_declines_an_image_the_track_does_not_reach():
    track = track_from((0.0, 0.0, 0.0, 5.0), (2.0, 20.0, 0.0, 5.0))

    assert gps.position_of(track, "front/000400.jpg", 0.5) is None
