import logging
from pathlib import Path


LOGGER_NAME = "rl_framework"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    
    if not name:
        return logging.getLogger(LOGGER_NAME)
    
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(
    log_file: Path,
    console: bool = True,
) -> tuple[logging.Logger, list[logging.Handler]]:
    
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
    return logger, handlers


def close_handlers(
    logger: logging.Logger,
    handlers: list[logging.Handler],
) -> None:
    
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
    handlers.clear()
