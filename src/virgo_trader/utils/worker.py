"""Qt worker helpers for running background tasks.

Defines a QRunnable-based worker and associated signals used by the dashboard to
run long operations without blocking the UI thread.
"""

import logging

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

logger = logging.getLogger(__name__)


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    Supported signals are:
    finished
        `object` data returned from processing, anything. Emits None on error.
    error
        `str` traceback.format_exc()
    progress
        `int` indicating % progress
    """

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    chart_update = pyqtSignal(dict)  # For thread-safe chart updates


class Worker(QRunnable):
    """
    一个通用的、可发出信号的Worker。
    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        # 从kwargs中提取特定的回调函数，以避免传递到不支持的函数中
        self.stop_flag_func = kwargs.pop("stop_flag_func", None)
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        """
        执行函数并处理信号。
        """
        try:
            # 准备要传递给目标函数的关键字参数
            callback_kwargs = {}
            if "progress_callback" in self.fn.__code__.co_varnames:
                callback_kwargs["progress_callback"] = self.signals
            if "status_callback" in self.fn.__code__.co_varnames:
                callback_kwargs["status_callback"] = self.signals.status.emit
            if self.stop_flag_func and "stop_flag_func" in self.fn.__code__.co_varnames:
                callback_kwargs["stop_flag_func"] = self.stop_flag_func

            result = self.fn(*self.args, **self.kwargs, **callback_kwargs)

        except Exception as e:
            import traceback

            error_str = traceback.format_exc()
            logger.exception("Worker execution failed: %s", e)
            self.signals.error.emit(error_str)
            self.signals.finished.emit(None)  # Emit finished with None on error
        else:
            self.signals.finished.emit(result)  # Emit finished with the result on success
