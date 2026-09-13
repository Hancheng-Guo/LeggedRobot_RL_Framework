from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from runners.utils import frames as frames_module
from runners.utils.frames import save_frames_to_video


def test_save_frames_chooses_format_and_suffix(tmp_path: Path) -> None:
    frames = [
        np.zeros((4, 6, 3), dtype=np.uint8),
        np.full((4, 6, 3), 255, dtype=np.uint8),
    ]

    output_paths = save_frames_to_video(
        frames,
        directory=tmp_path,
        file_name="rollout",
        fps=20.0,
    )

    assert output_paths == [tmp_path / "rollout.gif"]
    output_path = output_paths[0]
    assert output_path.is_file()
    with Image.open(output_path) as image:
        assert image.format == "GIF"
        assert image.n_frames == 2
        assert image.info["duration"] == 50


def test_save_frames_generates_file_name(tmp_path: Path) -> None:
    output_paths = save_frames_to_video(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        directory=tmp_path,
        fps=30.0,
    )

    assert len(output_paths) == 1
    output_path = output_paths[0]
    assert output_path.parent == tmp_path
    assert output_path.name.startswith("play_")
    assert output_path.suffix == ".gif"


def test_save_frames_rejects_suffix_in_file_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without a suffix"):
        save_frames_to_video(
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            directory=tmp_path,
            fps=30.0,
            file_name="rollout.mp4",
        )


def test_save_frames_writes_every_requested_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_save_mp4(
        frames: list[np.ndarray],
        output_path: Path,
        fps: float,
    ) -> None:
        output_path.touch()

    monkeypatch.setattr(frames_module, "_save_mp4", fake_save_mp4)
    output_paths = save_frames_to_video(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        directory=tmp_path,
        fps=30.0,
        file_name="rollout",
        formats=["mp4", "gif"],
    )

    assert output_paths == [
        tmp_path / "rollout.mp4",
        tmp_path / "rollout.gif",
    ]
    assert all(path.is_file() for path in output_paths)


@pytest.mark.parametrize("formats", [[], ["gif", "gif"], "avi"])
def test_save_frames_rejects_invalid_formats(
    tmp_path: Path,
    formats: Any,
) -> None:
    with pytest.raises(ValueError):
        save_frames_to_video(
            [np.zeros((2, 2, 3), dtype=np.uint8)],
            directory=tmp_path,
            fps=30.0,
            formats=formats,
        )
