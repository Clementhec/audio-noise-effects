import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging is enabled
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
        format_string: Custom format string. If None, uses default format

    Returns:
        Configured logger instance

    Example:
        >>> from utils.logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Default format: timestamp - logger name - level - message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Logger instance

    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Message")
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


# Pre-configured project logger for quick use
project_logger = setup_logger(
    name="sound-effects",
    level=logging.INFO,
    log_file=Path("logs/sound-effects.log"),
)
