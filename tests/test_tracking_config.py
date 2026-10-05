from runners.utils import TrackingSpec, parse_tracking_specs


def test_parse_tracking_specs_includes_plot_labels_and_limits() -> None:
    playback = [{
        "target": "command/custom",
        "measured": "state/custom_velocity[1]",
        "label": "Custom velocity (m/s)",
        "limits": [-3, 4],
    }]

    assert parse_tracking_specs(playback) == (
        TrackingSpec(
            "command/custom", "state/custom_velocity", 1,
            "Custom velocity (m/s)", (-3.0, 4.0),
        ),
    )
