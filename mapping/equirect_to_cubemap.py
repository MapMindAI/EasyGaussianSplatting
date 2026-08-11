#!/usr/bin/env python3
"""Convert a COLMAP EQUIRECTANGULAR reconstruction into a cube-map one.

gsplat only supports perspective/fisheye COLMAP camera models, not COLMAP's
EQUIRECTANGULAR model, so panoramic frames must be split into per-face
perspective views (a 90-degree-FOV cubemap) before training, with the
equirect camera poses reprojected into per-face poses to match.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

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


def face_basis(face):
    return (np.array(v, dtype=np.float64) for v in FACE_AXES[face])


def face_to_equirect_matrix(face):
    """Camera axes (x=right, y=down, z=forward) expressed in equirect-camera
    axes, as columns."""
    forward, up, right = face_basis(face)
    return np.stack([right, -up, forward], axis=1)


def face_rotation(face):
    """Rotation from equirect-camera axes to this face's camera axes."""
    # The inverse (transpose, since orthonormal) of face_to_equirect_matrix
    # rotates equirect-camera-frame vectors into this face's camera frame.
    return face_to_equirect_matrix(face).T


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
    return map_x, map_y


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ])


def rotmat_to_qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flatten()
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def ensure_txt_sparse_model(sparse_dir):
    if os.path.isfile(os.path.join(sparse_dir, "cameras.txt")):
        return
    subprocess.run(
        ["colmap", "model_converter",
         "--input_path", sparse_dir, "--output_path", sparse_dir, "--output_type", "TXT"],
        check=True,
    )


def iter_data_lines(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_txt(path):
    cameras = {}
    for line in iter_data_lines(path):
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(p) for p in parts[4:]],
        }
    return cameras


def read_images_txt(path):
    with open(path) as f:
        lines = [line for line in f if not line.startswith("#")]
    images = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        images.append({
            "image_id": int(parts[0]),
            "qvec": tuple(float(x) for x in parts[1:5]),
            "tvec": tuple(float(x) for x in parts[5:8]),
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
    return images


def read_points3d_txt(path):
    points = []
    for line in iter_data_lines(path):
        parts = line.split()
        points.append({
            "id": int(parts[0]),
            "xyz": tuple(float(x) for x in parts[1:4]),
            "rgb": tuple(int(x) for x in parts[4:7]),
            "error": float(parts[7]),
        })
    return points


def write_cameras_txt(path, face_size, camera_ids):
    # 90-degree FOV: tan(45 deg) == 1 spans the half-width of the face in NDC.
    focal = face_size / 2.0
    principal = face_size / 2.0
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(camera_ids)}\n")
        for camera_id in camera_ids.values():
            f.write(
                f"{camera_id} PINHOLE {face_size} {face_size} "
                f"{focal} {focal} {principal} {principal}\n"
            )


def write_images_txt(path, frames):
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}, mean observations per image: 0\n")
        for frame in frames:
            qw, qx, qy, qz = frame["qvec"]
            tx, ty, tz = frame["tvec"]
            f.write(
                f"{frame['image_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} "
                f"{frame['camera_id']} {frame['name']}\n"
            )
            f.write("\n")


def write_points3d_txt(path, points):
    # Tracks (2D-3D correspondences) reference the source equirect image IDs,
    # which don't correspond to these per-face images, so they're dropped
    # rather than remapped; gsplat's Parser only reads xyz/color/error for
    # initialization, not tracks (used only for optional per-image depth
    # supervision, off by default).
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for point in points:
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            f.write(f"{point['id']} {x} {y} {z} {r} {g} {b} {point['error']}\n")


def convert(reconstruction_dir, output_dir, face_size, faces):
    image_dir = os.path.join(reconstruction_dir, "images")
    sparse_dir = os.path.join(reconstruction_dir, "sparse", "0")
    ensure_txt_sparse_model(sparse_dir)

    cameras = read_cameras_txt(os.path.join(sparse_dir, "cameras.txt"))
    equirect_images = read_images_txt(os.path.join(sparse_dir, "images.txt"))
    points = read_points3d_txt(os.path.join(sparse_dir, "points3D.txt"))
    if not equirect_images:
        sys.exit(f"No registered images found in {sparse_dir}")

    equirect_camera = cameras[equirect_images[0]["camera_id"]]
    if equirect_camera["model"] != "EQUIRECTANGULAR":
        sys.exit(f"Expected an EQUIRECTANGULAR camera, got {equirect_camera['model']}")
    equirect_width, equirect_height = equirect_camera["width"], equirect_camera["height"]

    out_image_dir = os.path.join(output_dir, "images")
    out_sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_sparse_dir, exist_ok=True)

    remaps = {face: build_remap(face, face_size, equirect_width, equirect_height) for face in faces}
    camera_ids = {face: camera_id for camera_id, face in enumerate(faces, start=1)}
    face_rotations = {face: face_rotation(face) for face in faces}

    frames = []
    next_image_id = 1
    equirect_images.sort(key=lambda im: im["name"])
    for equirect_image in equirect_images:
        stem = os.path.splitext(equirect_image["name"])[0]
        equirect = cv2.imread(os.path.join(image_dir, equirect_image["name"]))
        if equirect is None:
            sys.exit(f"Could not read {equirect_image['name']} from {image_dir}")

        r_equirect = qvec_to_rotmat(equirect_image["qvec"])
        t_equirect = np.array(equirect_image["tvec"], dtype=np.float64)

        for face in faces:
            map_x, map_y = remaps[face]
            cube_face = cv2.remap(equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            name = f"{stem}_{face}.jpg"
            cv2.imwrite(os.path.join(out_image_dir, name), cube_face)

            r_face = face_rotations[face] @ r_equirect
            t_face = face_rotations[face] @ t_equirect
            frames.append({
                "image_id": next_image_id,
                "qvec": tuple(rotmat_to_qvec(r_face)),
                "tvec": tuple(t_face),
                "camera_id": camera_ids[face],
                "name": name,
            })
            next_image_id += 1

    write_cameras_txt(os.path.join(out_sparse_dir, "cameras.txt"), face_size, camera_ids)
    write_images_txt(os.path.join(out_sparse_dir, "images.txt"), frames)
    write_points3d_txt(os.path.join(out_sparse_dir, "points3D.txt"), points)
    print(f"Wrote {len(equirect_images)} frames x {len(faces)} faces to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reconstruction_dir",
                         help="colmap_reconstruct.sh-style dir containing images/ and sparse/0/")
    parser.add_argument("output_dir", help="directory to write the cube-map images/ and sparse/0/")
    parser.add_argument("--face-size", type=int, default=1024,
                         help="width/height in pixels of each cube face (default: 1024)")
    parser.add_argument("--faces", default="front,right,back,left,up,down",
                         help="comma-separated subset of front,right,back,left,up,down "
                              "(default: all six; drop 'down' if a tripod occludes the nadir)")
    args = parser.parse_args()
    faces = args.faces.split(",")
    unknown = set(faces) - set(FACE_AXES)
    if unknown:
        sys.exit(f"Unknown face(s): {', '.join(sorted(unknown))}")
    convert(args.reconstruction_dir, args.output_dir, args.face_size, faces)


if __name__ == "__main__":
    main()
