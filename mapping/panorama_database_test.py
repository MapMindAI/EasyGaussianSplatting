from pathlib import Path

from mapping.panorama_database import SampledVideo, expected_image_count


def test_expected_image_count_uses_all_sampled_frames_and_faces():
    videos = [
        SampledVideo(Path("first.mp4"), 0, 4, 0.5),
        SampledVideo(Path("second.mp4"), 4, 7, 0.5),
    ]

    assert expected_image_count(videos, ("front", "right", "up")) == 21
