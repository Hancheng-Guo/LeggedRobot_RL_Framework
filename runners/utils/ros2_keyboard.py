"""ROS 2 keyboard publisher and command source for human playback."""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
from collections.abc import Mapping
from typing import Any

from .human_ros2_config import HumanRos2Config, RosCommandMapping

PUBLISH_INTERVAL_MS = 50
COMMAND_TIMEOUT_SECONDS = 0.5
KEYBOARD_STARTUP_TIMEOUT_SECONDS = 60.0


class _KeyboardGears:
    def __init__(self, mappings: Mapping[str, RosCommandMapping]) -> None:
        self.mappings = mappings
        self.positions = {
            name: mapping.levels.index(0.0)
            for name, mapping in mappings.items()
        }
        self.held: set[str] = set()
        self.keys = {
            key: (name, direction)
            for name, mapping in mappings.items()
            for key, direction in (
                (mapping.positive_key, 1),
                (mapping.negative_key, -1),
            )
        }

    def press(self, key: str) -> bool:
        if key == "space":
            self.zero()
            return True
        if key not in self.keys or key in self.held:
            return False
        self.held.add(key)
        name, direction = self.keys[key]
        self.positions[name] = max(
            0,
            min(len(self.mappings[name].levels) - 1, self.positions[name] + direction),
        )
        return True

    def release(self, key: str) -> None:
        self.held.discard(key)

    def zero(self) -> None:
        for name, mapping in self.mappings.items():
            self.positions[name] = mapping.levels.index(0.0)

    def reset(self) -> None:
        self.zero()
        self.held.clear()

    def value(self, name: str) -> float:
        return self.mappings[name].levels[self.positions[name]]


def _keyboard_publisher(
    stop: Any,
    ready: Any,
    config: HumanRos2Config
) -> None:
    """Run the keyboard UI in its own process so simulation remains responsive."""

    context = None
    node = None
    window = None

    try:
        import tkinter as tk

        import rclpy  # pyright: ignore[reportMissingImports]
        from rclpy.context import Context  # pyright: ignore[reportMissingImports]
        from geometry_msgs.msg import Twist  # pyright: ignore[reportMissingImports]

        context = Context()
        context.init()
        node = rclpy.create_node("quadruped_keyboard", context=context)
        publisher = node.create_publisher(Twist, config.topic, 1)
        window = tk.Tk()
        window.title("Quadruped ROS 2 keyboard control")
        controls = tk.Frame(window)
        controls.pack(padx=24, pady=(24, 8))
        for row, (name, mapping) in enumerate(config.ros_commands.items()):
            tk.Label(controls, text=f"{name}:", anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 16),
            )
            tk.Label(controls, text=f"{mapping.positive_key.upper()} ↑").grid(
                row=row, column=1, padx=(0, 8),
            )
            tk.Label(controls, text="/").grid(row=row, column=2, padx=(0, 8))
            tk.Label(controls, text=f"{mapping.negative_key.upper()} ↓").grid(
                row=row, column=3,
            )
        tk.Label(window, text="Space zero · Close to exit").pack(
            padx=24, pady=(0, 24),
        )
        tk.Label(window, text="Command values", anchor="w").pack(fill="x", padx=24)
        values_label = tk.Label(
            window, justify="left", anchor="w", font=("Consolas", 11),
        )
        values_label.pack(fill="x", padx=24, pady=(0, 24))
        fixed_rows = [
            f"{name}: [{', '.join(f'{value:+.2f}' for value in values)}] (fixed)"
            for name, values in config.fixed_commands.items()
        ]
        gears = _KeyboardGears(config.ros_commands)
        pending_releases: dict[str, str] = {}

        def publish() -> None:
            message = Twist()
            current_rows = []
            for name, mapping in config.ros_commands.items():
                value = gears.value(name)
                vector_name, component = mapping.field.split(".")
                setattr(getattr(message, vector_name), component, value)
                current_rows.append(
                    f"{name}: {value:+.2f} "
                    f"({gears.positions[name] + 1}/{len(mapping.levels)})"
                )
            publisher.publish(message)
            values_label.configure(text="\n".join(current_rows + fixed_rows))

        def key_down(event: Any) -> None:
            key = event.keysym.lower()
            pending = pending_releases.pop(key, None)
            if pending is not None:
                window.after_cancel(pending)
            if gears.press(key):
                publish()

        def finish_release(key: str) -> None:
            pending_releases.pop(key, None)
            gears.release(key)

        def key_up(event: Any) -> None:
            key = event.keysym.lower()
            if key in gears.keys:
                pending = pending_releases.pop(key, None)
                if pending is not None:
                    window.after_cancel(pending)
                pending_releases[key] = window.after(50, finish_release, key)

        def zero_and_publish() -> None:
            for pending in pending_releases.values():
                window.after_cancel(pending)
            pending_releases.clear()
            gears.reset()
            publish()

        def close_window() -> None:
            zero_and_publish()
            window.quit()

        def tick() -> None:
            if stop.is_set():
                window.quit()
                return
            publish()
            window.after(PUBLISH_INTERVAL_MS, tick)

        window.bind("<KeyPress>", key_down)
        window.bind("<KeyRelease>", key_up)
        window.bind("<FocusOut>", lambda _event: zero_and_publish())
        window.protocol("WM_DELETE_WINDOW", close_window)
        ready.put(None)
        tick()
        window.mainloop()
        zero_and_publish()

    except BaseException as error:
        ready.put(f"{type(error).__name__}: {error}")
        raise

    finally:
        if window is not None:
            window.destroy()
        if node is not None:
            node.destroy_node()
        if context is not None and context.ok():
            context.shutdown()


class Ros2KeyboardSession:

    def __init__(
        self,
        config: HumanRos2Config
    ) -> None:

        try:
            import rclpy  # pyright: ignore[reportMissingImports]
            from rclpy.context import Context  # pyright: ignore[reportMissingImports]
            from rclpy.executors import SingleThreadedExecutor  # pyright: ignore[reportMissingImports]
            from geometry_msgs.msg import Twist  # pyright: ignore[reportMissingImports]
        except ImportError as error:
            raise ModuleNotFoundError(
                "Human ROS 2 playback requires rclpy and geometry_msgs "
                "in the selected Python environment."
            ) from error

        self._rclpy = rclpy
        self._config = config
        self._context = Context()
        self._context.init()
        self._node = None
        self._executor = None
        self._process = None
        self._stop = None
        self._values = {name: 0.0 for name in config.ros_commands}
        self._last_message_at: float | None = None

        try:
            self._node = rclpy.create_node("quadruped_playback", context=self._context)
            self._node.create_subscription(Twist, config.topic, self._on_message, 1)
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            process_context = mp.get_context("spawn")
            self._stop = process_context.Event()
            ready = process_context.Queue()
            self._process = process_context.Process(
                target=_keyboard_publisher, args=(self._stop, ready, config), daemon=True,
            )
            self._process.start()
            deadline = time.monotonic() + KEYBOARD_STARTUP_TIMEOUT_SECONDS
            while True:
                try:
                    wait_seconds = min(0.5, max(0.0, deadline - time.monotonic()))
                    error_message = ready.get(timeout=wait_seconds)
                    break
                except queue.Empty:
                    if not self._process.is_alive():
                        raise RuntimeError(
                            f"Keyboard window exited during startup with code {self._process.exitcode}."
                        )
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Keyboard window did not start within {KEYBOARD_STARTUP_TIMEOUT_SECONDS:g} seconds."
                        )
            if error_message is not None:
                raise RuntimeError(f"Keyboard window failed: {error_message}")

        except BaseException:
            self.close()
            raise

    def _on_message(self, message: Any) -> None:
        values: dict[str, float] = {}
        for name, mapping in self._config.ros_commands.items():
            vector_name, component = mapping.field.split(".")
            value = getattr(getattr(message, vector_name), component)
            if not math.isfinite(value):
                return
            values[name] = max(mapping.levels[0], min(mapping.levels[-1], value))
        self._values = values
        self._last_message_at = time.monotonic()

    def poll(self) -> None:
        if self._node is None or self._executor is None:
            raise RuntimeError("ROS 2 keyboard session is closed.")
        self._executor.spin_once(timeout_sec=0.0)
        if self._last_message_at is None or time.monotonic() - self._last_message_at > COMMAND_TIMEOUT_SECONDS:
            self._values = {name: 0.0 for name in self._config.ros_commands}

    def values(self) -> Mapping[str, float]:
        return self._values

    def is_open(self) -> bool:
        if self._process is None:
            return False
        if self._process.is_alive():
            return True
        if self._process.exitcode != 0:
            raise RuntimeError(
                f"Keyboard window exited with code {self._process.exitcode}."
            )
        return False

    def close(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._process is not None and self._process.pid is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
            self._process = None
        if self._node is not None:
            if self._executor is not None:
                self._executor.shutdown()
                self._executor = None
            self._node.destroy_node()
            self._node = None
        if self._context.ok():
            self._context.shutdown()
