import math
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Protocol, TypeAlias, cast
from PIL import Image

from .tracking_overlay import TrackingSpec, compose_tracking_frames


SUPPORTED_VIDEO_FORMATS = frozenset({"gif", "mp4"})
VideoFormat: TypeAlias = Literal["gif", "mp4"]
VideoFormats: TypeAlias = list[VideoFormat] | tuple[VideoFormat, ...]


class VideoWriter(Protocol):

    def append_data(self, image: np.ndarray) -> None: ...

    def close(self) -> None: ...


def save_frames_to_video(
    frames: list[np.ndarray],
    directory: Path,
    fps: float,
    file_name: str | None = None,
    formats: VideoFormat | VideoFormats = "gif",
    commands: list[tuple[float, ...]] | None = None,
    measured: list[tuple[float, ...]] | None = None,
    tracking_specs: tuple[TrackingSpec, ...] | None = None,
) -> list[Path]:
    """Compose tracking plots with frames, then save each requested format."""

    if not frames:
        raise ValueError("Cannot save an empty frame sequence.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("'fps' must be a finite value greater than 0.")
    if file_name is None:
        file_name = f"play_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if (
        not file_name
        or Path(file_name).name != file_name
        or Path(file_name).suffix
    ):
        raise ValueError(
            "'file_name' must be a non-empty file stem without a suffix."
        )

    requested_formats = (
        [formats]
        if isinstance(formats, str)
        else list(formats)
    )
    if not requested_formats:
        raise ValueError("'formats' cannot be empty.")
    if any(
        not isinstance(format_name, str)
        for format_name in requested_formats
    ):
        raise TypeError("Every video format must be a string.")

    normalized_formats = [
        format_name.lower()
        for format_name in requested_formats
    ]
    unsupported = set(normalized_formats) - SUPPORTED_VIDEO_FORMATS
    if unsupported:
        raise ValueError(
            f"Unsupported video formats: {sorted(unsupported)!r}. "
            f"Supported formats: {sorted(SUPPORTED_VIDEO_FORMATS)!r}."
        )
    if len(set(normalized_formats)) != len(normalized_formats):
        raise ValueError("'formats' cannot contain duplicates.")

    normalized_frames = _normalize_frames(frames)
    if (commands is None) != (measured is None):
        raise ValueError("'commands' and 'measured' must be provided together.")
    if commands is not None and measured is not None and (
        len(commands) != len(normalized_frames)
        or len(measured) != len(normalized_frames)
    ):
        raise ValueError("Tracking data must have one sample per frame.")
    if commands is not None and measured is not None and tracking_specs is None:
        raise ValueError("Tracking configuration is required with tracking data.")
    if commands is not None and measured is not None and tracking_specs is not None and any(
        len(command) != len(tracking_specs) or len(actual) != len(tracking_specs)
        for command, actual in zip(commands, measured)
    ):
        raise ValueError("Tracking configuration must match the data width.")
    directory.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    try:
        for format_name in normalized_formats:
            output_path = directory / f"{file_name}.{format_name}"
            composed_frames = (
                compose_tracking_frames(normalized_frames, commands, measured, tracking_specs, fps)
                if commands is not None and measured is not None and tracking_specs is not None
                else iter(normalized_frames)
            )
            if format_name == "gif":
                _save_gif(composed_frames, output_path, fps)
            else:
                _save_mp4(composed_frames, output_path, fps)
            output_paths.append(output_path)
    except Exception:
        for output_path in output_paths:
            if output_path.exists():
                output_path.unlink()
        raise

    return output_paths


def _normalize_frames(
    frames: list[np.ndarray]
) -> list[np.ndarray]:

    normalized: list[np.ndarray] = []
    expected_shape = frames[0].shape
    if len(expected_shape) != 3 or expected_shape[-1] not in (3, 4):
        raise ValueError("Frames must have shape [height, width, 3 or 4].")

    for frame in frames:
        if frame.shape != expected_shape:
            raise ValueError("All frames must have the same shape.")
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        normalized.append(frame.astype(np.uint8, copy=False))
    return normalized


def _save_gif(
    frames: Iterable[np.ndarray],
    output_path: Path,
    fps: float,
) -> None:

    images = [Image.fromarray(frame) for frame in frames]
    temporary_path = output_path.with_stem(output_path.stem + ".tmp")
    try:
        images[0].save(
            temporary_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=max(1, round(1000.0 / fps)),
            loop=0,
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _save_mp4(
    frames: Iterable[np.ndarray],
    output_path: Path,
    fps: float,
) -> None:

    temporary_path = output_path.with_stem(output_path.stem + ".tmp")
    try:
        writer = cast(
            VideoWriter,
            imageio.get_writer(
                str(temporary_path),
                mode="I",
                fps=fps,
                codec="libx264",
            ),
        )
        try:
            for frame in frames:
                writer.append_data(frame[..., :3])
        finally:
            writer.close()
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
