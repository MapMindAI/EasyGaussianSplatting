from mapping.dji.stitch import position_at, read_srt_positions


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

    positions = read_srt_positions(subtitles)

    assert len(positions) == 2
    assert positions[0].seconds == 0.0
    assert positions[0].altitude == -53.984
    assert position_at(positions, 0.49) == positions[1]
    assert position_at(positions, 0.1) == positions[0]
