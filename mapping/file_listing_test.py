from mapping.file_listing import files_under


def test_files_under_lists_nested_files_through_symlinked_folders(tmp_path):
    (tmp_path / "images" / "front").mkdir(parents=True)
    (tmp_path / "images" / "front" / "a.jpg").touch()
    (tmp_path / "images" / "b.jpg").touch()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "c.jpg").touch()
    (tmp_path / "images" / "linked").symlink_to(tmp_path / "elsewhere")

    assert sorted(files_under(tmp_path / "images")) == [
        tmp_path / "images" / "b.jpg",
        tmp_path / "images" / "front" / "a.jpg",
        tmp_path / "images" / "linked" / "c.jpg",
    ]


def test_files_under_a_missing_directory_is_empty(tmp_path):
    assert files_under(tmp_path / "missing") == []
