"""Parse and render configured playback tracking plots."""

import math
import re
import numpy as np
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterator
from PIL import Image, ImageDraw, ImageFont


STATE_COMPONENT = re.compile(r"^(state/[A-Za-z0-9_]+)\[(\d+)\]$")
SMOOTH_WINDOW = 25
RAW_ACTUAL = (125, 211, 252, 64)
SMOOTH_ACTUAL = "#38bdf8"


@dataclass(frozen=True)
class TrackingSpec:
    command_key: str
    state_key: str
    component: int
    label: str
    limits: tuple[float, float]


def parse_tracking_specs(playback: Sequence[Mapping[str, Any]]) -> tuple[TrackingSpec, ...]:
    specs: list[TrackingSpec] = []
    for plot in playback:
        command_key = plot.get("target")
        if not isinstance(command_key, str) or not command_key.startswith("command/"):
            raise ValueError(f"Invalid playback command key: {command_key!r}")
        state_ref = plot.get("measured")
        match = STATE_COMPONENT.fullmatch(state_ref) if isinstance(state_ref, str) else None
        if match is None:
            raise ValueError(f"Invalid playback state reference: {state_ref!r}")
        label = plot.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Invalid playback label for {command_key!r}.")
        limits = plot.get("limits")
        if (
            not isinstance(limits, list)
            or len(limits) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in limits)
            or not all(math.isfinite(value) for value in limits)
            or limits[0] >= limits[1]
        ):
            raise ValueError(f"Invalid playback limits for {command_key!r}.")
        specs.append(TrackingSpec(
            command_key, match.group(1), int(match.group(2)),
            label, (float(limits[0]), float(limits[1])),
        ))
    return tuple(specs)


def render_tracking_data_frame(
    frame_size: tuple[int, int],
    commands: list[tuple[float, ...]],
    measured: list[tuple[float, ...]],
    tracking_specs: tuple[TrackingSpec, ...],
) -> np.ndarray:
    """Render configured tracking plots for one playback frame."""
    source_width, source_height = frame_size
    display_height = round(source_height * 1.5)
    panel_width = round(source_width * 1.3)
    width, height = len(tracking_specs) * panel_width, display_height + 8
    canvas = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = title_font = ImageFont.load_default()
    actual_array = np.asarray(measured, dtype=float)
    raw_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    raw_draw = ImageDraw.Draw(raw_layer)
    smooth_lines: list[list[tuple[int, int]]] = []

    for axis, spec in enumerate(tracking_specs):
        label = spec.label
        low, high = spec.limits
        panel_left = axis * panel_width
        left, right = panel_left + 82, panel_left + panel_width - 24
        plot_top, plot_bottom = 96, height - 36
        draw.text((panel_left + 20, 20), label, fill="white", font=title_font)
        draw.rectangle((left, plot_top, right, plot_bottom), outline="#475569")
        zero_y = round(plot_bottom - (0 - low) / (high - low) * (plot_bottom - plot_top))
        draw.line((left, zero_y, right, zero_y), fill="#334155")
        draw.text((panel_left + 20, plot_top - 8), f"{high:g}", fill="#94a3b8", font=font)
        draw.text((panel_left + 20, plot_bottom - 10), f"{low:g}", fill="#94a3b8", font=font)

        def point(index: int, value: float) -> tuple[int, int]:
            x = left + round(index * (right - left) / max(1, len(commands) - 1))
            y = plot_bottom - round((value - low) / (high - low) * (plot_bottom - plot_top))
            return x, max(plot_top, min(plot_bottom, y))

        target_points = [point(i, row[axis]) for i, row in enumerate(commands)]
        raw_points = [point(i, row[axis]) for i, row in enumerate(measured)]
        if len(target_points) > 1:
            draw.line(target_points, fill="#f59e0b", width=3)
        if len(raw_points) > 1:
            raw_draw.line(raw_points, fill=RAW_ACTUAL, width=3)
        if measured:
            values = actual_array[:, axis]
            totals = np.convolve(values, np.ones(SMOOTH_WINDOW), mode="full")[:len(values)]
            counts = np.minimum(np.arange(1, len(values) + 1), SMOOTH_WINDOW)
            averages = totals / counts
            smooth_lines.append([point(i, float(value)) for i, value in enumerate(averages)])
            if commands:
                values = (
                    ("Target", commands[-1][axis]),
                    ("Actual", measured[-1][axis]),
                    ("Smoothed", averages[-1]),
                )
                max_magnitude = max(abs(low), abs(high))
                number_slot_width = max(
                    draw.textlength(f"{max_magnitude:+.2f}", font=font),
                    draw.textlength(f"{-max_magnitude:+.2f}", font=font),
                )
                column_left = panel_left + 20
                for name, value in values:
                    draw.text((column_left, 48), name, fill="#cbd5e1", font=font)
                    number = f"{value:+.2f}"
                    number_right = column_left + draw.textlength(name, font=font) + 8 + number_slot_width
                    number_x = number_right - draw.textlength(number, font=font)
                    draw.text((number_x, 48), number, fill="#cbd5e1", font=font)
                    column_left = number_right + 12
    canvas = Image.alpha_composite(canvas.convert("RGBA"), raw_layer).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for points in smooth_lines:
        if len(points) > 1:
            draw.line(points, fill=SMOOTH_ACTUAL, width=3)
    return np.asarray(canvas)


def compose_tracking_frame(
    frame: np.ndarray,
    data_frame: np.ndarray,
    elapsed_seconds: float,
) -> np.ndarray:
    """Place robot footage beside its tracking plots and draw their legend."""

    source = Image.fromarray(frame[..., :3])
    display_width = round(source.width * 1.5)
    display_height = round(source.height * 1.5)
    canvas = Image.new(
        "RGB", (display_width + data_frame.shape[1], data_frame.shape[0]), "#111827",
    )
    canvas.paste(source.resize((display_width, display_height), Image.Resampling.BICUBIC))
    canvas.paste(Image.fromarray(data_frame), (display_width, 0))
    legend_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(legend_layer)
    draw.rectangle((0, 0, display_width, 44), fill=(0, 0, 0, 112))
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.line((18, 22, 44, 22), fill="#f59e0b", width=4)
    draw.text((52, 12), "Target", fill="#e5e7eb", font=font)
    draw.line((126, 22, 152, 22), fill=RAW_ACTUAL, width=3)
    draw.text((160, 12), "Actual", fill="#e5e7eb", font=font)
    draw.line((238, 22, 264, 22), fill=SMOOTH_ACTUAL, width=4)
    draw.text((272, 12), f"Smooth ({SMOOTH_WINDOW})", fill="#e5e7eb", font=font)
    time_label = f"T +{elapsed_seconds:05.2f}"
    draw.text((display_width - 18 - font.getlength(time_label), 12), time_label,
              fill="#e5e7eb", font=font)
    return np.asarray(Image.alpha_composite(canvas.convert("RGBA"), legend_layer).convert("RGB"))


def compose_tracking_frames(
    frames: list[np.ndarray],
    commands: list[tuple[float, ...]],
    measured: list[tuple[float, ...]],
    tracking_specs: tuple[TrackingSpec, ...],
    fps: float,
) -> Iterator[np.ndarray]:
    """Render each plot from its samples and combine it with the matching frame."""
    for index, frame in enumerate(frames):
        data_frame = render_tracking_data_frame(
            (frame.shape[1], frame.shape[0]),
            commands[:index + 1],
            measured[:index + 1],
            tracking_specs,
        )
        yield compose_tracking_frame(frame, data_frame, index / fps)
