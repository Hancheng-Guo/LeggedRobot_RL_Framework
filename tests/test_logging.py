from __future__ import annotations

import logging

import pytest
from colorama import Back, Fore, Style

from utils.logging import ColoredFormatter, LOG_FORMAT


pytestmark = pytest.mark.core


@pytest.mark.parametrize(
    ("level", "color"),
    (
        (logging.DEBUG, Fore.LIGHTBLACK_EX),
        (logging.INFO, Fore.WHITE),
        (logging.WARNING, Fore.YELLOW),
        (logging.ERROR, Fore.RED),
        (logging.CRITICAL, Style.BRIGHT + Back.RED + Fore.WHITE),
    ),
)
def test_colored_formatter_colors_the_entire_record(
    level: int,
    color: str,
) -> None:
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname=__file__,
        lineno=1,
        msg="message",
        args=(),
        exc_info=None,
    )

    formatted = ColoredFormatter(LOG_FORMAT).format(record)

    assert formatted.startswith(color)
    assert formatted.endswith(Style.RESET_ALL)
    assert " | message" in formatted
