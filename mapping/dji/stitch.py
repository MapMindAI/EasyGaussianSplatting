"""Export DJI panorama frames as six GPS-tagged pinhole faces."""

import argparse
import bisect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from ..panorama_database import DEFAULT_FACES, build_database, image_name


@dataclass(frozen=True)
class SrtPosition:
    seconds: float
    latitude: float
    longitude: float
    altitude: float


def read_srt_positions(path):
    positions = []
    for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig")):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        timestamp = re.match(r"(\d+):(\d+):(\d+),(\d+)\s+-->", lines[1])
        latitude = re.search(r"\[latitude:\s*([-+\d.]+)\]", block)
        longitude = re.search(r"\[longitude:\s*([-+\d.]+)\]", block)
        altitude = re.search(r"abs_alt:\s*([-+\d.]+)", block)
        if not all((timestamp, latitude, longitude, altitude)):
            raise ValueError(f"Missing time or GPS in SRT entry {lines[0]} of {path}")
        hours, minutes, seconds, milliseconds = map(int, timestamp.groups())
        positions.append(SrtPosition(
            hours * 3600 + minutes * 60 + seconds + milliseconds / 1000,
            float(latitude.group(1)), float(longitude.group(1)),
            float(altitude.group(1)),
        ))
    if not positions:
        raise ValueError(f"No GPS positions found in {path}")
    return positions


def position_at(positions, seconds):
    times = [position.seconds for position in positions]
    index = bisect.bisect_left(times, seconds)
    return min(
        (positions[i] for i in (index - 1, index) if 0 <= i < len(positions)),
        key=lambda position: abs(position.seconds - seconds),
    )


def write_gps_exif(image_paths, position):
    command = [
        "exiftool", "-overwrite_original", "-n",
        f"-GPSLatitude={abs(position.latitude)}",
        f"-GPSLatitudeRef={'N' if position.latitude >= 0 else 'S'}",
        f"-GPSLongitude={abs(position.longitude)}",
        f"-GPSLongitudeRef={'E' if position.longitude >= 0 else 'W'}",
        f"-GPSAltitude={abs(position.altitude)}",
        f"-GPSAltitudeRef={0 if position.altitude >= 0 else 1}",
        *(str(path) for path in image_paths),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def export_images(workspace_path, frame_rate=2.0, face_size=None):
    workspace_path = Path(workspace_path)
    videos = sorted(workspace_path.glob("*.mp4"))
    if len(videos) != 1:
        raise ValueError(f"Expected one MP4 in {workspace_path}, found {len(videos)}")
    srt_path = videos[0].with_suffix(".SRT")
    if not srt_path.is_file():
        raise FileNotFoundError(f"Missing DJI GPS subtitles: {srt_path}")
    positions = read_srt_positions(srt_path)
    capture = cv2.VideoCapture(str(videos[0]))
    try:
        frame_rate_of_video = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or frame_rate_of_video <= 0:
            raise ValueError(f"Could not read video timing from {videos[0]}")
        duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate_of_video
    finally:
        capture.release()
    if abs(duration - positions[-1].seconds) > 1.0:
        raise ValueError(
            f"{videos[0]} lasts {duration:.1f} s but {srt_path} ends at "
            f"{positions[-1].seconds:.1f} s"
        )
    video = build_database(videos, workspace_path, frame_rate, face_size, DEFAULT_FACES)[0]
    image_directory = workspace_path / "images"
    for frame_index in range(video.start_frame_index, video.stop_frame_index):
        position = position_at(positions, frame_index * video.seconds_per_frame)
        write_gps_exif(
            [image_directory / image_name(face, frame_index) for face in DEFAULT_FACES],
            position,
        )
    return video.stop_frame_index * len(DEFAULT_FACES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="directory containing one MP4 and matching SRT")
    parser.add_argument("--frame-rate", type=float, default=2.0)
    parser.add_argument("--face-size", type=int)
    args = parser.parse_args()
    count = export_images(args.workspace, args.frame_rate, args.face_size)
    print(f"Wrote GPS EXIF to {count} cube-face images in {args.workspace / 'images'}")


if __name__ == "__main__":
    main()
