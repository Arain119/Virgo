"""Qt-friendly logging integration.

Provides logging handlers that forward formatted records to the UI and utilities
to redirect stdout/stderr into Qt signals.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from .paths import RESEARCH_LOG_PATH, ensure_dir


class _LogSignalEmitter(QObject):
    """A minimal QObject wrapper used only for exposing PyQt signals."""

    log_received = pyqtSignal(str, str)  # (levelname, formatted_message)


class QLogHandler(logging.Handler):
    """
    A logging.Handler that forwards formatted log records to the UI via a PyQt signal.

    Notes on design:
    - This class intentionally does NOT inherit QObject.
    - Qt may delete QObject wrappers during application shutdown, while the logging module
      still iterates over handlers in `atexit`. Keeping the handler as a pure-Python
      logging.Handler avoids "wrapped C/C++ object has been deleted" errors at exit.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__()
        self._emitter = _LogSignalEmitter(parent)

    @property
    def log_received(self):
        """Expose the Qt signal for `handler.log_received.connect(...)`."""

        return self._emitter.log_received

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < self.level:
            return

        msg = self.format(record)
        try:
            self._emitter.log_received.emit(record.levelname, msg)
        except RuntimeError:
            # The Qt object may already be deleted during shutdown.
            pass


def setup_logging(qt_handler: QLogHandler) -> None:
    """Configure the root logging system (UI signal handler + console + file)."""

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    qt_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    ensure_dir(RESEARCH_LOG_PATH.parent)
    file_handler = logging.FileHandler(RESEARCH_LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Remove existing handlers deterministically to avoid duplicate logs.
    # Close them to release file locks on Windows.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: S110
            # Best-effort cleanup; avoid raising during logging reconfiguration / shutdown.
            pass

    logger.addHandler(qt_handler)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


class StreamRedirector(QObject):
    """Redirect stdout/stderr to a Qt signal for display in the UI."""

    message_written = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.buffer = ""

    def write(self, text: str) -> None:
        # Filter empty lines / noise.
        if not text.strip():
            return

        self.buffer += text
        if "\n" in self.buffer:
            self.message_written.emit(self.buffer.strip())
            self.buffer = ""

    def flush(self) -> None:
        if self.buffer.strip():
            self.message_written.emit(self.buffer.strip())
            self.buffer = ""
