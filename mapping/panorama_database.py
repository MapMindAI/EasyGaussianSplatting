"""Build a COLMAP rig database of cube-map views from a panorama video.

Every sampled equirectangular frame is reprojected into 90-degree-FOV pinhole
cube faces and registered as one COLMAP frame of a single rig: the front face
is the rig's reference sensor, and each remaining face is mounted on it by a
pure rotation, since all faces are rendered about the panorama's optical
centre. Mapping the faces as a rig leaves global SfM one pose to solve per
panorama instead of one per face, and holds the faces' relative orientations
at their exact values.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from pycolmap import logging

# forward, up, right, in equirect camera axes (+X right, +Y up, +Z forward).
# `right` is chosen so that (right, -up, forward) is right-handed -- i.e. a
# proper rotation, matching the standard (x=right, y=down, z=forward) pinhole
# camera frame -- which face_rotation() relies on for a valid camera pose.
FACE_AXES = {
    "front": ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    "right": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "back": ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    "left": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "up": ((0, 1, 0), (0, 0, -1), (-1, 0, 0)),
    "down": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
}

# Every face. The nadir shows whoever is carrying the rig, which person masking
# takes out of training, and the ground it also sees is texture the mapper can
# match on.
DEFAULT_FACES = ("front", "right", "back", "left", "up", "down")

# The rig's reference sensor: every other face's sensor_from_rig is expressed
# relative to this one.
REFERENCE_FACE = "front"


def face_to_equirect_matrix(face):
    """Camera axes (x=right, y=down, z=forward) expressed in equirect-camera
    axes, as columns."""
    forward, up, right = (np.array(v, dtype=np.float64) for v in FACE_AXES[face])
    return np.stack([right, -up, forward], axis=1)


def face_rotation(face):
    """Rotation from equirect-camera axes to this face's camera axes."""
    # The inverse (transpose, since orthonormal) of face_to_equirect_matrix
    # rotates equirect-camera-frame vectors into this face's camera frame.
    return face_to_equirect_matrix(face).T


# Gravity in the rig frame, which is the reference face's camera frame. The
# panorama's own axes put +Y up, so rotating its down axis into that face gives
# the direction an upright capture falls in -- whichever face is the reference.
RIG_DOWN = face_rotation(REFERENCE_FACE) @ np.array([0.0, -1.0, 0.0])


def sensor_from_rig(face):
    """This face's pose in the rig frame, which is the reference face's camera
    frame. The faces share the panorama's optical centre, so the mounting is a
    pure rotation."""
    rotation = face_rotation(face) @ face_rotation(REFERENCE_FACE).T
    return pycolmap.Rigid3d(np.hstack([rotation, np.zeros((3, 1))]))


def build_remap(face, face_size, equirect_width, equirect_height):
    to_equirect = face_to_equirect_matrix(face)
    u, v = np.meshgrid(np.arange(face_size), np.arange(face_size))
    ndc_x = 2.0 * (u + 0.5) / face_size - 1.0
    ndc_y = 2.0 * (v + 0.5) / face_size - 1.0
    ndc = np.stack([ndc_x, ndc_y, np.ones_like(ndc_x)], axis=-1)

    directions = ndc @ to_equirect.T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    theta = np.arctan2(directions[..., 0], directions[..., 2])
    phi = np.arcsin(np.clip(directions[..., 1], -1.0, 1.0))

    map_x = ((theta / (2 * np.pi) + 0.5) * equirect_width).astype(np.float32)
    map_y = ((0.5 - phi / np.pi) * equirect_height).astype(np.float32)
    # Fixed-point maps: cv2.remap's fast path, and half the memory of the
    # float32 pair (8 MB per face at a 1024 face size).
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)


def face_camera(camera_id, face_size):
    # 90-degree FOV: tan(45 deg) == 1 spans the half-width of the face in NDC.
    camera = pycolmap.Camera.create_from_model_name(
        camera_id, "PINHOLE", face_size / 2.0, face_size, face_size
    )
    # The focal length comes from the reprojection rather than a calibration,
    # so it is exact; global SfM warns and falls back to estimating focal
    # lengths from the view graph unless it is told that.
    camera.has_prior_focal_length = True
    return camera


def default_face_size(video_path):
    """A quarter of the panorama's width, which is what a 90-degree face needs
    to hold the equirectangular frame's angular resolution: the full 360 spans
    `width` pixels, so one face's 90 spans a quarter of them."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        sys.exit(f"Could not open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()
    if width <= 0:
        sys.exit(f"Could not read the frame width of {video_path}")
    return width // 4


def image_name(face, frame_index):
    """The database name of one cube face of one sampled panorama."""
    return f"{face}/{frame_index:06d}.jpg"


def parse_image_name(name):
    """The face and panorama index that `image_name` encoded."""
    face, _, index = name.partition("/")
    return face, int(Path(index).stem)


def sampling_step(video_path, frame_rate):
    """Video frames between the samples below, and the video's own frame rate.

    Timing a sampled frame -- which is what pairs it with a GPS fix -- needs
    both, so the sampling is derived here rather than in the loop.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        sys.exit(f"Could not open {video_path}")
    rate = capture.get(cv2.CAP_PROP_FPS) or frame_rate
    capture.release()
    return max(1, round(rate / frame_rate)), rate


def iter_panorama_frames(video_path, frame_rate):
    """Yields every `frame_rate`-th frame per second of the video."""
    step, _ = sampling_step(video_path, frame_rate)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        sys.exit(f"Could not open {video_path}")

    index = 0
    while capture.grab():
        if index % step == 0:
            ok, frame = capture.retrieve()
            if not ok:
                break
            yield frame
        index += 1
    capture.release()


def build_database(video_path, workspace_path, frame_rate, face_size, faces):
    """Writes `<workspace_path>/images/<face>/` and a `database.db` holding one
    camera per face, the rig mounting them, and one frame per panorama. A
    `face_size` of None takes `default_face_size(video_path)`.

    Returns without touching anything if the database already holds images:
    later stages write their own output into that same file, so rebuilding it
    would throw their work away.
    """
    workspace_path = Path(workspace_path)
    database_path = workspace_path / "database.db"
    if database_path.exists():
        with pycolmap.Database.open(database_path) as database:
            if database.num_images():
                logging.info(f"{database_path} already holds images, keeping it")
                return

    if face_size is None:
        face_size = default_face_size(video_path)

    images_dir = workspace_path / "images"
    for face in faces:
        (images_dir / face).mkdir(parents=True, exist_ok=True)
    database_path.unlink(missing_ok=True)

    cameras = {face: face_camera(index, face_size) for index, face in enumerate(faces, start=1)}
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(cameras[REFERENCE_FACE].sensor_id)
    for face in faces:
        if face != REFERENCE_FACE:
            rig.add_sensor(cameras[face].sensor_id, sensor_from_rig(face))

    logging.info(
        f"Database settings: frame_rate={frame_rate}, face_size={face_size}, "
        f"faces={','.join(faces)}"
    )

    remaps = {}
    image_id = 0
    with pycolmap.Database.open(database_path) as database:
        for camera in cameras.values():
            database.write_camera(camera, use_camera_id=True)
        database.write_rig(rig, use_rig_id=True)

        for frame_index, panorama in enumerate(iter_panorama_frames(video_path, frame_rate)):
            if not remaps:
                height, width = panorama.shape[:2]
                remaps = {
                    face: build_remap(face, face_size, width, height) for face in faces
                }
                logging.info(f"Panorama frames are {width}x{height}")

            frame = pycolmap.Frame()
            frame.frame_id = frame_index + 1
            frame.rig_id = rig.rig_id
            for face in faces:
                map_x, map_y = remaps[face]
                cv2.imwrite(
                    str(images_dir / image_name(face, frame_index)),
                    cv2.remap(
                        panorama,
                        map_x,
                        map_y,
                        cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP,
                    ),
                )

                image_id += 1
                image = pycolmap.Image(
                    name=image_name(face, frame_index),
                    camera_id=cameras[face].camera_id,
                    image_id=image_id,
                )
                database.write_image(image, use_image_id=True)
                frame.add_data_id(image.data_id)
            database.write_frame(frame, use_frame_id=True)

    if image_id == 0:
        sys.exit(f"Extracted no frames from {video_path}")
    logging.info(
        f"Registered {image_id // len(faces)} panorama frames x {len(faces)} faces "
        f"in {database_path}"
    )


def parse_faces(value):
    faces = value.split(",")
    unknown = set(faces) - set(FACE_AXES)
    if unknown:
        sys.exit(f"Unknown face(s): {', '.join(sorted(unknown))}")
    if REFERENCE_FACE not in faces:
        sys.exit(f"The face list must include '{REFERENCE_FACE}', the rig's reference sensor")
    return faces
