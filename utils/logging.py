import logging
from dataclasses import dataclass, field
from pathlib import Path


LOGGER_NAME = "rl_framework"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


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
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    for handler in handlers:
        logger.addHandler(handler)
    return LoggingSession(logger=logger, handlers=handlers)
