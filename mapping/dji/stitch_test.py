from mapping.dji import stitch
from mapping.panorama_database import DEFAULT_FACES, SampledVideo


def test_srt_positions_follow_subtitle_times_and_absolute_altitude(tmp_path):
    subtitles = tmp_path / "capture.SRT"
    subtitles.write_text(
        "1\n00:00:00,000 --> 00:00:00,033\n"
        "2026-10-26 12:59:46.281\n"
        "[latitude: 31.578604] [longitude: 120.296219] "
        "[rel_alt: 11.500 abs_alt: -53.984]\n\n"
        "2\n00:00:00,500 --> 00:00:00,533\n"
        "2026-10-26 12:59:46.781\n"
        "[latitude: 31.578700] [longitude: 120.296100] "
        "[rel_alt: 12.000 abs_alt: -53.484]\n"
    )

    positions = stitch.read_srt_positions(subtitles)

    assert len(positions) == 2
    assert positions[0].seconds == 0.0
    assert positions[0].altitude == -53.984
    assert stitch.position_at(positions, 0.49) == positions[1]
    assert stitch.position_at(positions, 0.1) == positions[0]


def test_exports_all_paired_videos_with_each_srt_timed_from_its_start(tmp_path, monkeypatch, capsys):
    for name in ("a.MP4", "a.SRT", "b.mp4", "b.srt", "unpaired.mp4"):
        (tmp_path / name).touch()

    class Capture:
        def isOpened(self):
            return True

        def get(self, property_id):
            return 2 if property_id == stitch.cv2.CAP_PROP_FPS else 2

        def release(self):
            pass

    monkeypatch.setattr(stitch.cv2, "VideoCapture", lambda path: Capture())
    monkeypatch.setattr(stitch, "read_srt_positions", lambda path: [
        stitch.SrtPosition(0.0, 1.0 if path.stem == "a" else 2.0, 3.0, 4.0),
        stitch.SrtPosition(0.5, 1.5 if path.stem == "a" else 2.5, 3.0, 4.0),
    ])

    def build_database(videos, workspace, frame_rate, face_size, faces):
        assert [path.name for path in videos] == ["a.MP4", "b.mp4"]
        assert faces == DEFAULT_FACES
        return [
            SampledVideo(videos[0], 0, 2, 0.5),
            SampledVideo(videos[1], 2, 4, 0.5),
        ]

    monkeypatch.setattr(stitch, "build_database", build_database)
    tagged = []
    monkeypatch.setattr(
        stitch, "write_gps_exif",
        lambda paths, position: tagged.append((paths, position)),
    )

    count = stitch.export_images(tmp_path)

    assert count == 4 * len(DEFAULT_FACES)
    assert [position.latitude for _, position in tagged] == [1.0, 1.5, 2.0, 2.5]
    assert [paths[0].name for paths, _ in tagged] == [
        "000000.jpg", "000001.jpg", "000002.jpg", "000003.jpg",
    ]
    output = capsys.readouterr().out
    assert "Processing a.MP4 with a.SRT" in output
    assert "Processing b.mp4 with b.srt" in output
    assert "begin frame ID 1, expected frames added 2, total frames now 2" in output
    assert "begin frame ID 3, expected frames added 2, total frames now 4" in output
