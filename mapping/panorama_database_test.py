from pathlib import Path
from types import SimpleNamespace

from mapping import panorama_database
from mapping.panorama_database import SampledVideo, expected_image_count


def test_expected_image_count_uses_all_sampled_frames_and_faces():
    videos = [
        SampledVideo(Path("first.mp4"), 0, 4, 0.5),
        SampledVideo(Path("second.mp4"), 4, 7, 0.5),
    ]

    assert expected_image_count(videos, ("front", "right", "up")) == 21


def test_cuda_decoding_selects_on_the_gpu_before_downloading():
    command = panorama_database.decode_command("pano.mp4", 15, "yuv420p10le", use_cuda=True)

    assert command[command.index("-hwaccel_output_format") + 1] == "cuda"
    assert command[command.index("-vf") + 1] == (
        "select=not(mod(n\\,15)),hwdownload,format=p010le"
    )
    assert command[-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def test_cpu_decoding_only_selects():
    command = panorama_database.decode_command("pano.mp4", 15, "yuv420p", use_cuda=False)

    assert "-hwaccel" not in command
    assert command[command.index("-vf") + 1] == "select=not(mod(n\\,15))"


def test_the_cuda_probe_decodes_one_frame_on_the_gpu(monkeypatch):
    commands = []
    monkeypatch.setattr(
        panorama_database.subprocess, "run",
        lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0),
    )

    assert panorama_database.cuda_decodes("pano.mp4", 15, "yuv420p")
    command = commands[0]
    assert command[command.index("-frames:v") + 1] == "1"
    assert command[command.index("-hwaccel_output_format") + 1] == "cuda"
