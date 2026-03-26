"""Small logging utilities used across the project."""

import logging

MARKDOWN_LOGGERS = [
    "markdown",
    "markdown.extensions",
    "markdown.blockprocessors",
    "markdown.extensions.tables",
    "markdown.extensions.fenced_code",
    "markdown.extensions.codehilite",
    "markdown.extensions.extra",
    "markdown.extensions.toc",
    "markdown.extensions.attr_list",
    "markdown.extensions.def_list",
    "markdown.extensions.md_in_html",
    "markdown.extensions.footnotes",
    "markdown.extensions.abbr",
]


def suppress_markdown_logging(level: int = logging.ERROR) -> None:
    """Reduce markdown-related log noise without global monkey patches."""
    for logger_name in MARKDOWN_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False
