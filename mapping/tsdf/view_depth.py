"""Display rendered cube-face depth maps on one shared colour scale."""
import argparse
from pathlib import Path

import numpy as np

from .common import depth_path


def cube_faces(depth_directory):
    """Return each face directory that holds depth maps, with its sorted frames."""
    faces = {}
    for face in sorted(path for path in Path(depth_directory).iterdir() if path.is_dir()):
        frames = sorted(frame.stem for frame in face.glob("*.npy"))
        if frames:
            faces[face.name] = frames
    if not faces:
        raise FileNotFoundError(f"No cube-face depth maps under {depth_directory}")
    return faces


def visible_depth(depth):
    """Blank the unrendered pixels so they read as missing, not as zero distance."""
    return np.where(depth > 0, depth, np.nan)


class DepthBrowser:
    """Which face and frame the viewer is on, and how the keys move between them."""

    def __init__(self, faces, face=None):
        self.faces = faces
        self.names = list(faces)
        if face is not None and face not in faces:
            raise KeyError(f"{face} holds no depth maps; try one of {self.names}")
        self.face_index = self.names.index(face) if face else 0
        self.frame_index = 0

    @property
    def face(self):
        return self.names[self.face_index]

    @property
    def frame(self):
        return self.faces[self.face][self.frame_index]

    def step_frame(self, delta):
        self.frame_index = (self.frame_index + delta) % len(self.faces[self.face])

    def step_face(self, delta):
        self.face_index = (self.face_index + delta) % len(self.names)
        self.frame_index = min(self.frame_index, len(self.faces[self.face]) - 1)


def view_depths(depth_directory, face=None, maximum_depth=20.0):
    """Open an interactive window over a directory of rendered depth maps."""
    import matplotlib
    import matplotlib.pyplot as pyplot

    depth_directory = Path(depth_directory)
    browser = DepthBrowser(cube_faces(depth_directory), face)
    depth = np.empty((1, 1))

    figure, axes = pyplot.subplots(figsize=(9, 8))
    figure.canvas.manager.set_window_title(f"Depth: {depth_directory}")
    image = axes.imshow(
        depth,
        cmap=matplotlib.colormaps["turbo"].with_extremes(bad="0.15"),
        vmin=0.0,
        vmax=maximum_depth,
        interpolation="nearest",
    )
    figure.colorbar(image, ax=axes).set_label("depth (m)")
    axes.set_xlabel("←/→ frame    ↑/↓ face    q quit")
    axes.set_xticks([])
    axes.set_yticks([])

    def format_coord(x, y):
        column, row = int(round(x)), int(round(y))
        if not (0 <= row < depth.shape[0] and 0 <= column < depth.shape[1]):
            return ""
        distance = depth[row, column]
        return f"({column}, {row})  " + (
            f"{distance:.2f} m" if distance > 0 else "no depth"
        )

    axes.format_coord = format_coord

    def draw():
        nonlocal depth
        depth = np.load(depth_path(depth_directory, f"{browser.face}/{browser.frame}"))
        image.set_data(visible_depth(depth))
        image.set_extent((-0.5, depth.shape[1] - 0.5, depth.shape[0] - 0.5, -0.5))
        height, width = depth.shape
        rendered = np.count_nonzero(depth > 0) / depth.size
        axes.set_title(
            f"{browser.face}/{browser.frame}    {width}x{height}    "
            f"{rendered:.1%} rendered"
        )
        figure.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "left"):
            browser.step_frame(1 if event.key == "right" else -1)
        elif event.key in ("up", "down"):
            browser.step_face(-1 if event.key == "up" else 1)
        else:
            return
        draw()

    figure.canvas.mpl_connect("key_press_event", on_key)
    draw()
    pyplot.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_directory", type=Path)
    parser.add_argument("--face", help="cube face to open on, e.g. front")
    parser.add_argument("--maximum-depth", type=float, default=20.0)
    arguments = parser.parse_args()
    view_depths(arguments.depth_directory, arguments.face, arguments.maximum_depth)


if __name__ == "__main__":
    main()
