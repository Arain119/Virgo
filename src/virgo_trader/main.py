"""Main Qt application entrypoint and orchestration.

Wires together the dashboard UI, background workers, training loop, model
management, and logging integration.
"""

import sys
import os

# Suppress TensorFlow oneDNN INFO messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Suppress TensorFlow INFO and WARNING messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")

import json
import logging
import subprocess
from datetime import datetime
from PyQt6.QtCore import QObject, QThreadPool, pyqtSlot
from PyQt6.QtWidgets import QApplication, QMessageBox

SSE50_POOL_BASELINE = "510050.SH"
from .data.stock_pools import get_stock_pool_codes
from .utils.sb3_compat import infer_ppo_model_spaces
from .utils.paths import (
    PROJECT_ROOT,
    BEST_PARAMS_PATH,
    MODELS_DIR,
    PROGRESS_FILE,
    STUDY_DB_PATH,
    LEGACY_BEST_PARAMS_PATH,
    LEGACY_PROGRESS_FILE,
    LEGACY_STUDY_DB_PATH,
    find_existing_best_params_path,
    is_user_model_path,
    list_model_zip_paths,
    resolve_model_zip_path,
)

# Use relative imports
from .train import train
from .ui.dashboard import ModernDashboard as RealTimeDashboard
from .ui.styled_message_box import StyledMessageBox
from .data.data_fetcher import get_stock_data
from .simulation.sim_trader import run_simulation_for_worker
from .utils.q_logger import setup_logging, QLogHandler, StreamRedirector
from .utils.logging_utils import suppress_markdown_logging
from .utils.worker import Worker
from .utils.database_manager import init_db, get_sessions_for_model, get_episodes_for_session


class MainApp(QObject):
    def __init__(self):
        super().__init__()

        init_db()  # 初始化数据库

        self.threadpool = QThreadPool()
        self.training_worker = None
        self.stop_requested = False

        self.full_kline_data = None
        self.current_loaded_episodes = {}  # 用于存储从数据库加载的回合数据

        self.dashboard = RealTimeDashboard()

        # 设置标准日志记录
        self.log_handler = QLogHandler(self)
        self.log_handler.setLevel(logging.INFO)  # Set default level
        self.log_handler.log_received.connect(self.append_log_message)
        self.setup_logging_system(self.log_handler)

        # 重定向 stdout 和 stderr
        self.redirector = StreamRedirector(self)
        self.redirector.message_written.connect(self.append_raw_log_message)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

        self.connect_signals()

        logging.info(f"Application started. Max threads: {self.threadpool.maxThreadCount()}.")
        if hasattr(self.dashboard, "model_list_widget"):
            self.refresh_models_list()

    def connect_signals(self):
        # Note: integrated_train_button is now handled by dashboard's own start_integrated_training method
        # The old training logic is kept for compatibility but not connected to any button
        self.dashboard.stop_train_button.clicked.connect(self.stop_training_worker)

        # 模型管理页面
        if hasattr(self.dashboard, "refresh_models_button"):
            self.dashboard.refresh_models_button.clicked.connect(self.refresh_models_list)
        if hasattr(self.dashboard, "delete_model_button"):
            self.dashboard.delete_model_button.clicked.connect(self.delete_selected_model)
        if hasattr(self.dashboard, "view_history_button"):
            self.dashboard.view_history_button.clicked.connect(
                self.load_sessions_for_selected_model
            )
        if hasattr(self.dashboard, "session_list_widget"):
            self.dashboard.session_list_widget.currentItemChanged.connect(
                self.load_episodes_for_selected_session
            )
        if hasattr(self.dashboard, "episode_combo_box"):
            self.dashboard.episode_combo_box.currentIndexChanged.connect(
                self.display_selected_episode
            )

        # 回测页面
        self.dashboard.start_backtest_button.clicked.connect(self.start_backtest_worker)
        self.dashboard.back_to_setup_button.clicked.connect(self.back_to_backtest_setup)
        self.dashboard.btn_backtest.clicked.connect(self.populate_backtest_model_list)

        # 日志页面
        self.dashboard.log_level_combo.currentTextChanged.connect(self.on_log_level_changed)

        # 设置页面
        self.dashboard.clear_opt_history_button.clicked.connect(self.clear_optimization_history)

    def on_log_level_changed(self, level_str):
        """根据UI选择更改日志处理器的级别"""
        level_map = {
            "ALL": logging.DEBUG,  # Show all logs from DEBUG upwards
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = level_map.get(level_str, logging.INFO)
        self.log_handler.setLevel(level)
        logging.info(f"Log display level set to {level_str}")

    def _fetch_reference_data(self, stock_codes, start_date, end_date):
        """Try sequentially to fetch k-line data for the provided stock list."""
        codes = stock_codes or []
        for code in codes:
            if not code:
                continue
            data = get_stock_data(code, start_date, end_date)
            if data is not None and not data.empty:
                return data, code
        return None, None

    def _run_optimization_and_training(
        self, train_params, dashboard, stop_flag_func, progress_callback=None
    ):
        """
        Handles the logic for conditional hyperparameter optimization and final training.
        This function will be run in a separate worker thread.
        """

        def emit_status(message):
            if progress_callback and hasattr(progress_callback, "status"):
                progress_callback.status.emit(message)

        try:
            best_params_path = find_existing_best_params_path() or BEST_PARAMS_PATH

            # Check if best_params.json exists
            if not best_params_path.exists():
                logging.info("best_params.json not found. Starting hyperparameter optimization...")
                emit_status("正在进行超参数优化...")

                try:
                    # Run optimize.py as a subprocess
                    agent_type_arg = train_params.get("agent_type", "multiscale_cnn")
                    command = [
                        sys.executable,
                        "-m",
                        "virgo_trader.training.optimize",
                        "--agent_type",
                        agent_type_arg,
                    ]
                    # Command is constructed by the application and executed without a shell.
                    optimize_process = subprocess.run(  # noqa: S603
                        command,
                        capture_output=True,
                        text=True,
                        check=True,  # Raise an exception for non-zero exit codes
                    )
                    logging.info("optimize.py output:\n" + optimize_process.stdout)
                    if optimize_process.stderr:
                        logging.warning("optimize.py errors/warnings:\n" + optimize_process.stderr)
                    logging.info("Hyperparameter optimization completed successfully.")
                    emit_status("超参数优化完成。")
                except subprocess.CalledProcessError as e:
                    logging.error(f"Hyperparameter optimization failed: {e}")
                    emit_status("超参数优化失败。")
                    return False  # Indicate failure
                except Exception as e:
                    logging.error(f"An unexpected error occurred during optimization: {e}")
                    emit_status("超参数优化时发生未知错误。")
                    return False  # Indicate failure
            else:
                logging.info("best_params.json found. Skipping hyperparameter optimization.")
                emit_status("使用现有最佳参数。")

            # Generate a unique model name for the final trained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_params["model_name"] = f"PPO_Final_Model_{timestamp}"

            logging.info(
                f"Starting final model training with model name: {train_params['model_name']}"
            )
            emit_status("正在训练最终模型...")

            # Call the train function (which now reads best_params.json automatically)
            train(train_params, dashboard, progress_callback, stop_flag_func)

            logging.info("Final model training completed.")
            emit_status("最终模型训练完成。")
            return True  # Indicate success

        except Exception as e:
            logging.error(
                f"An error occurred during the combined optimization/training process: {e}"
            )
            emit_status(f"训练/优化过程出错: {e}")
            return False  # Indicate failure

    def start_training_worker(self):
        self.stop_requested = False
        try:
            train_params = self.dashboard.get_train_parameters()
        except ValueError as e:
            StyledMessageBox.critical(self.dashboard, "无效输入", str(e))
            return

        self.dashboard.kline_chart.clear_all()

        # 准备K线图背景
        try:
            self.full_kline_data, reference_code = self._fetch_reference_data(
                train_params.get("stock_pool"), train_params["start_date"], train_params["end_date"]
            )
            if self.full_kline_data is None:
                raise ValueError(
                    "Failed to fetch k-line data for any symbol in the selected股票池。"
                )
            logging.info("K线背景使用标的: %s", reference_code)
            if self.full_kline_data.empty:
                raise ValueError("Failed to fetch k-line data.")
            self.dashboard.kline_chart.plot_background(self.full_kline_data)
        except Exception as e:
            StyledMessageBox.critical(self.dashboard, "数据错误", str(e))
            return

        self.dashboard.integrated_train_button.setEnabled(False)
        self.dashboard.stop_train_button.setEnabled(True)

        # Change this line to call the new combined function
        self.training_worker = Worker(
            self._run_optimization_and_training,
            train_params,
            self.dashboard,
            stop_flag_func=lambda: self.stop_requested,
        )
        self.training_worker.signals.status.connect(self.update_status_label)
        self.training_worker.signals.progress.connect(self.update_progress_bar)
        self.training_worker.signals.finished.connect(self.on_training_finished)
        self.training_worker.signals.error.connect(self.on_worker_error)
        self.training_worker.signals.chart_update.connect(
            self.on_chart_update_live
        )  # 连接到新的实时更新槽

        self.threadpool.start(self.training_worker)

    @pyqtSlot(dict)
    def on_chart_update_live(self, update_data):
        """只在训练期间更新K线图，不存储任何东西"""
        if hasattr(self.dashboard, "on_chart_update_live"):
            self.dashboard.on_chart_update_live(update_data)
            return
        self.dashboard.kline_chart.update_overlays(
            trades=update_data.get("trades"),
            portfolio_history=update_data.get("portfolio_history"),
        )

    def load_sessions_for_selected_model(self):
        """当用户点击'查看训练记录'时，加载所选模型的训练会话"""
        model_list_widget = getattr(self.dashboard, "model_list_widget", None)
        session_list_widget = getattr(self.dashboard, "session_list_widget", None)
        episode_combo_box = getattr(self.dashboard, "episode_combo_box", None)
        if not model_list_widget or not session_list_widget or not episode_combo_box:
            logging.debug(
                "Model history widgets missing; skipping load_sessions_for_selected_model."
            )
            StyledMessageBox.warning(
                self.dashboard, "界面不支持", "当前版本未提供模型历史查看功能。"
            )
            return
        selected_items = model_list_widget.selectedItems()
        if not selected_items:
            StyledMessageBox.warning(self.dashboard, "未选择模型", "请选择一个模型。")
            return

        model_name = selected_items[0].text().replace(".zip", "")
        logging.info(f"Loading sessions for model: {model_name}")

        session_list_widget.clear()
        episode_combo_box.clear()
        self.current_loaded_episodes.clear()

        sessions = get_sessions_for_model(model_name)
        if not sessions:
            StyledMessageBox.information(
                self.dashboard, "无记录", f"未找到模型 '{model_name}' 的训练记录。"
            )
            return

        for session in sessions:
            # 添加会话到列表，并存储其ID
            item_text = f"Session {session['session_id']} @ {session['start_time']}"
            session_list_widget.addItem(item_text)
            # 将实际的session id存储在item的用户数据中
            last_item = session_list_widget.item(session_list_widget.count() - 1)
            last_item.setData(1, session["session_id"])  # 使用Qt.UserRole的整数键

    def load_episodes_for_selected_session(self, current_item, previous_item):
        """当用户选择一个会话时，加载该会话的所有回合"""
        if not current_item:
            return
        # Unused: `previous_item` is part of the Qt signal signature (current, previous).
        _ = previous_item
        episode_combo_box = getattr(self.dashboard, "episode_combo_box", None)
        if episode_combo_box is None:
            logging.debug("Episode combo box missing; skipping load_episodes_for_selected_session.")
            return

        session_id = current_item.data(1)
        logging.info(f"Loading episodes for session: {session_id}")

        episode_combo_box.clear()
        self.current_loaded_episodes.clear()

        episodes = get_episodes_for_session(session_id)
        if not episodes:
            return

        for episode in episodes:
            episode_num = episode["episode_number"]
            episode_combo_box.addItem(f"回合 {episode_num}", userData=episode_num)
            self.current_loaded_episodes[episode_num] = episode

    def display_selected_episode(self, index):
        """在用户选择特定回合时更新K线图"""
        episode_combo_box = getattr(self.dashboard, "episode_combo_box", None)
        model_list_widget = getattr(self.dashboard, "model_list_widget", None)
        if episode_combo_box is None or model_list_widget is None:
            logging.debug("History widgets missing; skipping display_selected_episode.")
            return
        episode_num = episode_combo_box.itemData(index)
        if episode_num is None:
            return
        episode_data = self.current_loaded_episodes.get(episode_num)
        if not episode_data:
            return
        try:
            trades = episode_data["trades"]
            if not trades or trades.empty:
                raise ValueError("选定回合没有交易数据")

            self.dashboard.kline_chart.update_overlays(
                trades=trades, portfolio_history=episode_data.get("portfolio_history", [])
            )

            first_trades = trades.iloc[:1]
            last_trades = trades.iloc[-1:]
            if first_trades.empty or last_trades.empty:
                raise ValueError("交易数据为空，无法确定日期范围")

            if len(first_trades.index) == 0 or len(last_trades.index) == 0:
                raise ValueError("交易数据索引无效")

            first_trade_date = first_trades.index.min().strftime("%Y%m%d")
            last_trade_date = last_trades.index.max().strftime("%Y%m%d")

            selected_model_items = model_list_widget.selectedItems()
            if not selected_model_items:
                return
            model_name = selected_model_items[0].text().replace(".zip", "")

            session_id = episode_data.get("session_id")
            session_info = None
            if session_id is not None:
                session_info = next(
                    (
                        s
                        for s in get_sessions_for_model(model_name)
                        if s["session_id"] == session_id
                    ),
                    None,
                )
            if not session_info:
                return

            params = json.loads(session_info["train_parameters"])
            self.full_kline_data, reference_code = self._fetch_reference_data(
                params.get("stock_pool"), first_trade_date, last_trade_date
            )
            if self.full_kline_data is None:
                raise ValueError("Failed to fetch k-line data for history view.")
            logging.info("历史回放采用标的: %s", reference_code)
            self.dashboard.kline_chart.plot_background(self.full_kline_data)
        except Exception as e:
            logging.error(f"Error displaying episode {episode_num}: {e}")
            StyledMessageBox.critical(self.dashboard, "回放错误", f"无法显示回放: {e}")

    def on_training_finished(self, result):
        logging.info("Worker thread finished.")
        self.dashboard.progress_label.setText("Idle")
        self.dashboard.progress_bar.setValue(0)
        self.dashboard.integrated_train_button.setEnabled(True)
        self.dashboard.stop_train_button.setEnabled(False)

        if self.stop_requested:
            StyledMessageBox.warning(self.dashboard, "已停止", "训练已被用户停止。")
        elif result:
            StyledMessageBox.information(self.dashboard, "已完成", "模型训练完成并已保存。")
            self.refresh_models_list()
        else:  # result is None, indicating an error
            StyledMessageBox.warning(self.dashboard, "训练失败", "训练过程失败，请检查日志。")

        self.training_worker = None
        self.stop_requested = False

    def refresh_models_list(self):
        model_list_widget = getattr(self.dashboard, "model_list_widget", None)
        if model_list_widget is None:
            logging.debug("Model list widget not available; skipping refresh.")
            return
        model_list_widget.clear()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            models = [p.name for p in list_model_zip_paths()]
            model_list_widget.addItems(models)
            logging.info(f"Found {len(models)} models.")
        except Exception as e:
            logging.error(f"Failed to refresh models list: {e}")

    def delete_selected_model(self):
        model_list_widget = getattr(self.dashboard, "model_list_widget", None)
        if model_list_widget is None:
            StyledMessageBox.warning(self.dashboard, "未找到列表", "当前界面不再提供模型管理功能。")
            return
        selected_items = model_list_widget.selectedItems()
        if not selected_items:
            StyledMessageBox.warning(self.dashboard, "未选择模型", "请选择一个要删除的模型。")
            return

        model_name = selected_items[0].text()
        resolved_model_path = MODELS_DIR / model_name
        try:
            resolved_model_path = resolve_model_zip_path(model_name)
        except FileNotFoundError:
            pass
        model_path = resolved_model_path

        if not is_user_model_path(model_path):
            StyledMessageBox.warning(
                self.dashboard, "禁止删除", "该模型为内置只读模型，无法删除。"
            )
            return
        if not model_path.exists():
            StyledMessageBox.warning(self.dashboard, "模型缺失", f"未找到模型文件：{model_name}")
            return

        msg_box = StyledMessageBox(self.dashboard)
        msg_box.set_icon(QMessageBox.Icon.Question)
        try:
            relative_path = str(model_path.relative_to(PROJECT_ROOT))
        except ValueError:
            relative_path = str(model_path)
        msg_box.set_text(f"确定要删除 '{model_name}' 吗？\n文件: {relative_path}\n此操作无法撤销。")
        msg_box.setWindowTitle("确认删除")
        yes_button = msg_box.add_button("确认", QMessageBox.ButtonRole.YesRole)
        no_button = msg_box.add_button("取消", QMessageBox.ButtonRole.NoRole)
        msg_box.set_default_button(no_button)

        msg_box.exec()

        if msg_box.clicked_button() == yes_button:
            try:
                model_path.unlink()
                StyledMessageBox.information(
                    self.dashboard, "成功", f"模型 '{model_name}' 已删除。"
                )
                self.refresh_models_list()
            except Exception as e:
                StyledMessageBox.critical(self.dashboard, "错误", f"删除模型失败: {e}")

    def stop_training_worker(self):
        if self.training_worker:
            logging.info("Stop training requested by user.")
            self.stop_requested = True
            self.dashboard.stop_train_button.setEnabled(False)

    @pyqtSlot(str)
    def update_status_label(self, status):
        self.dashboard.progress_label.setText(status)

    @pyqtSlot(int)
    def update_progress_bar(self, value):
        self.dashboard.progress_bar.setValue(value)

    @pyqtSlot(str, str)
    def append_log_message(self, levelname, message):
        color_map = {
            "INFO": "#f8f8f2",
            "WARNING": "#f1fa8c",
            "ERROR": "#ff5555",
            "CRITICAL": "#ff5555",
        }
        color = color_map.get(levelname, "#f8f8f2")
        self.dashboard.log_display.append(f'<font color="{color}">{message}</font>')

    @pyqtSlot(str)
    def append_raw_log_message(self, message):
        # 使用灰色显示来自stdout/stderr的原始文本
        self.dashboard.log_display.append(f'<font color="#888888">{message}</font>')

    @pyqtSlot(str)
    def on_worker_error(self, error_message):
        logging.error(f"Worker error: {error_message}")
        StyledMessageBox.critical(self.dashboard, "线程错误", error_message)
        if self.training_worker:
            self.on_training_finished()

    def populate_backtest_model_list(self):
        """当用户切换到回测页面时，填充模型列表"""
        refresh_func = getattr(self.dashboard, "refresh_backtest_model_list", None)
        if callable(refresh_func):
            refresh_func()
            return
        self.dashboard.backtest_model_list.clear()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            models = [p.name for p in list_model_zip_paths()]
            self.dashboard.backtest_model_list.addItems(models)
        except Exception as e:
            logging.error(f"Failed to populate backtest models list: {e}")

    def start_backtest_worker(self):
        selected_items = self.dashboard.backtest_model_list.selectedItems()
        if not selected_items:
            StyledMessageBox.warning(
                self.dashboard, "未选择模型", "请从列表中选择一个模型进行回测。"
            )
            return

        model_name = selected_items[0].text().replace(".zip", "")
        start_date = self.dashboard.backtest_start_date.date().toString("yyyyMMdd")
        end_date = self.dashboard.backtest_end_date.date().toString("yyyyMMdd")

        # 从训练参数中获取股票代码和其他必要信息
        # 注意：这里我们假设模型名称与某个训练会话相关联，以便获取参数。
        # 一个更鲁棒的实现可能会将元数据与模型文件一起保存。
        try:
            # --- 动态获取 window_size ---
            model_path = resolve_model_zip_path(model_name)
            spaces_info = infer_ppo_model_spaces(model_path)
            window_size = int(spaces_info.window_size)
            # window_size inferred from checkpoint metadata (no model object loaded)
            logging.info("Inferred window_size=%s from model '%s'.", window_size, model_name)

            sessions = get_sessions_for_model(model_name)
            params = {}
            if sessions:
                latest_session = sorted(sessions, key=lambda x: x["start_time"], reverse=True)[0]
                params = json.loads(latest_session["train_parameters"])

            commission_rate = float(params.get("commission_rate") or 0.0003)
            slippage = float(params.get("slippage") or 0.0001)

            inferred_agent_type = "multiscale_cnn"
            name_upper = model_name.upper()
            if "TRANSFORMER" in name_upper:
                inferred_agent_type = "transformer"
            elif "CROSS_ATTENTION" in name_upper:
                inferred_agent_type = "cross_attention"
            elif "MULTISCALE_CNN" in name_upper:
                inferred_agent_type = "multiscale_cnn"
            agent_type = (params.get("agent_type") or inferred_agent_type).lower()

            stock_codes = [code for code in (params.get("stock_pool") or []) if code]
            if not stock_codes and "SSE50_POOL" in name_upper:
                stock_codes = [
                    code
                    for code in get_stock_pool_codes("sse50", use_live=False)
                    if code != SSE50_POOL_BASELINE
                ]
            if not stock_codes:
                raise ValueError("Unable to resolve stock_pool for backtest.")
            reference_code = stock_codes[0]
            is_multi_asset = len(stock_codes) > 1
            stock_pool_payload = stock_codes if is_multi_asset else [reference_code]
            base_symbol = reference_code if not is_multi_asset else SSE50_POOL_BASELINE

            backtest_params = {
                "stock_code": reference_code,
                "base_symbol": base_symbol,
                "stock_pool": stock_pool_payload,
                "start_date": start_date,
                "end_date": end_date,
                "commission_rate": commission_rate,
                "slippage": slippage,
                "window_size": window_size,
                "model_name": model_name,
                "model_path": str(model_path),
                "agent_type": agent_type,
            }

            sentiment_dataset = str(params.get("sentiment_dataset") or "").strip()
            if sentiment_dataset and os.path.exists(sentiment_dataset):
                backtest_params["sentiment_dataset"] = sentiment_dataset
        except Exception as e:
            StyledMessageBox.critical(self.dashboard, "参数错误", f"无法为回测加载参数: {e}")
            return

        self.dashboard.stacked_widget.setCurrentWidget(self.dashboard.backtest_results_page)
        self.dashboard.backtest_current_page = self.dashboard.backtest_results_page
        self.dashboard.backtest_status_label.setText(f"正在为 {model_name} 进行回测...")
        self.dashboard.backtest_kline_chart.clear_all()

        # 使用通用工作线程运行模拟
        backtest_worker = Worker(run_simulation_for_worker, backtest_params)
        backtest_worker.signals.finished.connect(self.on_backtest_finished)
        # error signal is now mainly for logging, finished(None) handles the UI flow
        backtest_worker.signals.error.connect(
            lambda e: logging.error(f"Backtest worker error: {e}")
        )
        self.threadpool.start(backtest_worker)

    def on_backtest_finished(self, results):
        """当回测完成时，在图表上显示结果"""
        if results is None:
            self.dashboard.backtest_status_label.setText("回测失败。请检查日志。")
            StyledMessageBox.critical(
                self.dashboard, "回测失败", "回测过程中发生错误，请查看日志输出获取详细信息。"
            )
            return

        logging.info("Backtest finished. Displaying results.")

        try:
            # 使用新的统一显示函数
            self.dashboard.display_backtest_results(results)
        except Exception as e:
            logging.error(f"Error displaying backtest results: {e}")
            StyledMessageBox.critical(self.dashboard, "显示错误", f"无法显示回测结果: {e}")
            self.dashboard.backtest_status_label.setText("显示结果时出错。")

    def back_to_backtest_setup(self):
        """切换回回测设置页面"""
        self.dashboard.stacked_widget.setCurrentWidget(self.dashboard.backtest_setup_page)
        self.dashboard.backtest_current_page = self.dashboard.backtest_setup_page

    def clear_optimization_history(self):
        """Deletes all optimization-related history files."""
        files_to_delete = [
            STUDY_DB_PATH,
            PROGRESS_FILE,
            BEST_PARAMS_PATH,
            LEGACY_STUDY_DB_PATH,
            LEGACY_PROGRESS_FILE,
            LEGACY_BEST_PARAMS_PATH,
        ]

        existing_files = [path for path in files_to_delete if path.exists()]

        if not existing_files:
            StyledMessageBox.information(self.dashboard, "无记录", "没有需要清除的优化记录。")
            return

        msg_box = StyledMessageBox(self.dashboard)
        msg_box.set_icon(QMessageBox.Icon.Question)
        display_paths = [
            str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
            for path in existing_files
        ]
        files_str = "\n".join(display_paths)
        msg_box.set_text(
            f"您确定要永久删除所有优化记录吗？\n以下文件将被删除:\n{files_str}\n此操作无法撤销。"
        )
        msg_box.setWindowTitle("确认清除记录")
        yes_button = msg_box.add_button("确认", QMessageBox.ButtonRole.YesRole)
        no_button = msg_box.add_button("取消", QMessageBox.ButtonRole.NoRole)
        msg_box.set_default_button(no_button)

        msg_box.exec()

        if msg_box.clicked_button() == yes_button:
            deleted_count = 0
            errors = []
            for path in existing_files:
                try:
                    path.unlink()
                    logging.info(f"Successfully deleted optimization history file: {path}")
                    deleted_count += 1
                except Exception as e:
                    error_msg = f"Failed to delete optimization history file '{path}': {e}"
                    logging.error(error_msg)
                    errors.append(error_msg)

            if not errors:
                StyledMessageBox.information(
                    self.dashboard, "成功", f"优化记录已成功清除 ({deleted_count}个文件)。"
                )
            else:
                StyledMessageBox.critical(
                    self.dashboard, "错误", "删除部分历史文件时出错:\n" + "\n".join(errors)
                )

    def show(self):
        self.dashboard.show()

    def setup_logging_system(self, handler):
        setup_logging(handler)
        suppress_markdown_logging()


def main():
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
