import sys
from typing import Optional

import loguru
from loguru import logger as default_logger


def setup_logging(
    verbose: bool, log_filename: Optional[str] = None, logger=None
) -> "loguru.Logger":
    """
    Set up logging configuration.

    It removes any existing loggers, sets the log format, and adds handlers for both standard output
    and file logging (if log_file is provided).
    The log level is determined based on the verbose parameter. The log format includes the timestamp,
    log level, module name, function name, line number, and log message.

    The log messages are colorized for better readability on the console.

    AArgs:
        `verbose` (bool): Whether to enable verbose logging. If True, the log level will be set to DEBUG. If False, the log level will be set to INFO.
        `log_file` (Optional[str]): Optional parameter specifying the log filename. If provided, logs will be written to the specified file in addition to the standard output. Defaults to None.
        `logger` (Optional[loguru.Logger]): Optional parameter allowing injection of a logger instance. If not provided, the default logger is used.

    Returns:
        `loguru.Logger`: The configured logger object.
    """
    logger = logger or default_logger
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    level = "DEBUG" if verbose else "INFO"

    # stdout handler
    logger.add(
        sys.stdout,
        # filter=ydl_debug_log_filter(level="DEBUG"),
        format=log_format,
        level=level,
        colorize=True,
    )
    # file handler
    if log_filename:
        logger.add(
            f"{log_filename}.log",
            format=log_format,
            level=level,
            rotation="1 week",
            retention="2 weeks",
        )

    logger.info("Logging setup complete")
    return logger
