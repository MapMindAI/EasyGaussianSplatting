#!/usr/bin/env python3
"""Extract frames from a video into a directory of numbered JPEGs."""
import argparse
import sys

import cv2


def extract_images(video_path, image_dir, frame_rate):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Could not open {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or frame_rate
    step = max(1, round(video_fps / frame_rate))

    index = written = 0
    while True:
        if index % step != 0:
            if not cap.grab():
                break
            index += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(f"{image_dir}/{written:06d}.jpg", frame)
        written += 1
        index += 1

    if written == 0:
        sys.exit(f"Extracted no frames from {video_path}")
    print(f"Extracted {written} frames from {video_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path")
    parser.add_argument("image_dir")
    parser.add_argument("--frame-rate", type=float, default=2.0,
                         help="frames per second to sample from the video (default: 2.0)")
    args = parser.parse_args()
    extract_images(args.video_path, args.image_dir, args.frame_rate)


if __name__ == "__main__":
    main()
