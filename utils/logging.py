import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from colorama import Back, Fore, Style, just_fix_windows_console


LOGGER_NAME = "rl_framework"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


class ColoredFormatter(logging.Formatter):
    """Color an entire console record according to its severity."""

    LEVEL_COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Style.BRIGHT + Back.RED + Fore.WHITE,
    }

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.LEVEL_COLORS.get(record.levelno, "")
        if not color:
            return message
        return f"{color}{message}{Style.RESET_ALL}"


def get_logger(name: str | None = None) -> logging.Logger:
    
    if not name:
        return logging.getLogger(LOGGER_NAME)
    
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


@dataclass
class LoggingSession:
    logger: logging.Logger
    handlers: list[logging.Handler] = field(default_factory=list)
    _closed: bool = False

    def flush(self) -> None:
        for handler in self.handlers:
            handler.flush()

    def close(self) -> None:
        if self._closed:
            return
        for handler in self.handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
        self.handlers.clear()
        self._closed = True


def configure_logging(
    log_file: Path,
    console: bool = True,
) -> LoggingSession:
    
    logger = get_logger()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [file_handler]

    if console:
        console_handler = logging.StreamHandler()
        color_enabled = (
            "NO_COLOR" not in os.environ
            and console_handler.stream.isatty()
        )
        if color_enabled:
            just_fix_windows_console()
            console_handler.setFormatter(ColoredFormatter(LOG_FORMAT))
        else:
            console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    for handler in handlers:
        logger.addHandler(handler)
    return LoggingSession(logger=logger, handlers=handlers)
