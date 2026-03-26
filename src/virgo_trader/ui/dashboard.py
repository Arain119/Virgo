"""Main dashboard window for Virgo Trader.

Hosts training controls, charts, log output, and the embedded assistant chat
panel used to inspect results and interact with the system.
"""

import datetime
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyqtgraph as pg
from PyQt6 import sip
from PyQt6.QtCore import (
    QDate,
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    QThread,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..data.data_fetcher import get_stock_data
from ..data.sentiment_loader import list_sentiment_datasets
from ..data.stock_pools import get_stock_pool_codes, list_stock_pools
from ..train import train
from ..utils.paths import (
    BEST_PARAMS_PATH,
    LEGACY_BEST_PARAMS_PATH,
    LEGACY_PROGRESS_FILE,
    LEGACY_STUDY_DB_PATH,
    MODELS_DIR,
    PROGRESS_FILE,
    PROJECT_ROOT,
    STUDY_DB_PATH,
    find_existing_best_params_path,
    find_existing_progress_file,
    is_user_model_path,
    list_model_zip_paths,
)
from ..utils.performance_metrics import calculate_performance_metrics
from ..utils.worker import Worker
from .chat.chat_widget import ChatWidget
from .kline_chart import KLineChartWidget
from .styled_message_box import StyledMessageBox

logger = logging.getLogger(__name__)

CONFIG_DIR = PROJECT_ROOT / "config"
USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"
try:
    from ..news import config as news_config

    NEWS_EXPORT_DIR = news_config.EXPORT_DIR
except Exception:
    NEWS_EXPORT_DIR = PROJECT_ROOT / "news" / "data" / "exports"
NEWS_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CLOUD_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_CLOUD_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
DEFAULT_CLOUD_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")


class ExternalProcessWorker(QThread):
    """
    A worker thread to run long-running external commands
    and emit their output in real-time.
    """

    output_received = pyqtSignal(str)
    process_finished = pyqtSignal(int)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self.process = None

    def run(self):
        try:
            # Command is constructed by the application and executed without a shell.
            self.process = subprocess.Popen(  # noqa: S603
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
            for line in iter(self.process.stdout.readline, ""):
                self.output_received.emit(line)

            self.process.stdout.close()
            return_code = self.process.wait()
            self.process_finished.emit(return_code)
        except Exception as e:
            self.output_received.emit(f"Failed to start process: {e}\n")
            self.process_finished.emit(-1)

    def stop(self):
        if self.process and self.process.poll() is None:
            try:
                if sys.platform == "win32":
                    # Send CTRL+C event on Windows
                    self.process.send_signal(signal.CTRL_C_EVENT)
                else:
                    # Send SIGINT on Unix-like systems
                    self.process.send_signal(signal.SIGINT)

                # Wait for a moment to allow graceful shutdown
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # If it doesn't shut down gracefully, force terminate
                self.process.terminate()
            except Exception as e:
                # Fallback to terminate on any other error
                logger.exception("Error sending signal: %s", e)
                self.process.terminate()
            finally:
                self.process.wait()


class ModernDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virgo Trader")
        # Set a more reasonable default size
        self.setGeometry(100, 100, 1100, 700)
        self.setMinimumSize(800, 600)  # Set a minimum size

        # --- Enable Acrylic/Blur Effect ---
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.container_layout = QHBoxLayout(self.central_widget)
        self.container_layout.setContentsMargins(0, 0, 0, 0)

        self.background_frame = QFrame(self.central_widget)
        self.background_frame.setObjectName("backgroundFrame")
        self.container_layout.addWidget(self.background_frame)

        self.main_layout = QHBoxLayout(self.background_frame)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- Custom Window Frame Logic ---
        self.border_width = 5
        self.resizing = False
        self.moving = False
        self.resize_edge = None
        self.old_pos = None

        # --- User Config ---
        self.user_config_path = USER_CONFIG_PATH
        self.user_config = {}
        self.load_user_config()
        self.stock_pool_options = list_stock_pools()
        pool_meta = next((opt for opt in self.stock_pool_options if opt["key"] == "sse50"), None)
        pool_label = pool_meta["label"] if pool_meta else "上证50股票池"
        pool_codes = list(get_stock_pool_codes("sse50"))
        self.stock_pool_config = {
            "mode": "pool",
            "codes": pool_codes,
            "key": "sse50",
            "label": pool_label,
        }
        self.current_optimization_code = pool_codes[0]

        self.create_main_content()
        # Initialize backtest state before creating nav bar
        self.backtest_current_page = self.backtest_setup_page
        self.create_nav_bar()

        self.main_layout.addWidget(self.nav_bar)
        self.main_layout.addWidget(self.main_content)

        self.apply_stylesheet()

        self._setup_animations()
        self.stacked_widget.setCurrentWidget(self.dashboard_page)

        # --- Worker Thread ---
        self.worker_thread = None
        self.worker = None

        # --- In-Process Worker Management ---
        self.threadpool = QThreadPool()
        self.training_worker_obj = None
        self.stop_requested = False
        self.sentiment_worker_obj = None
        self.sentiment_job_running = False

        # --- Training State Management ---
        self.training_stage = None  # None, 'optimization', 'training'
        self.integrated_training_active = False

        # --- Data Cache for Integrated Training ---
        self.cached_data = None  # Store data from optimization to reuse in training
        self.cached_raw_data = None  # Store raw data from optimization
        self.current_backtest_results = None

        # --- Optimization Monitor Timer ---
        self.optimization_timer = QTimer(self)
        self.optimization_timer.timeout.connect(self.update_optimization_monitor)

    def load_user_config(self):
        if os.path.exists(self.user_config_path):
            try:
                with open(self.user_config_path, "r", encoding="utf-8") as f:
                    self.user_config = json.load(f)
            except json.JSONDecodeError:
                self.user_config = {"nickname": "Guest", "email": ""}
        else:
            self.user_config = {"nickname": "Guest", "email": ""}
            Path(self.user_config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.user_config_path, "w", encoding="utf-8") as f:
                json.dump(self.user_config, f, indent=4)

        # Backward-compat: treat legacy placeholder email as "unset" to avoid accidental notifications.
        legacy_email = self.user_config.get("email")
        if isinstance(legacy_email, str) and legacy_email.strip().lower() == "example@mail.com":
            self.user_config["email"] = ""

        # Update UI elements that depend on user config
        if hasattr(self, "nickname_input"):
            self.nickname_input.setText(self.user_config.get("nickname", "Guest"))
            self.email_input.setText(self.user_config.get("email", ""))
            # Load notification preferences, defaulting to False if not present
            self.notify_on_optimization_finish_checkbox.setChecked(
                self.user_config.get("notify_on_optimization_finish", False)
            )
            self.notify_on_training_finish_checkbox.setChecked(
                self.user_config.get("notify_on_training_finish", False)
            )
        if hasattr(self, "user_nickname_label"):
            self.update_user_bar()

    def save_user_config(self, notify=True):
        nickname = self.nickname_input.text().strip()
        email = self.email_input.text().strip()

        if not nickname and not email:
            self.user_config["nickname"] = "Guest"
            self.user_config["email"] = ""
        else:
            self.user_config["nickname"] = nickname
            self.user_config["email"] = email

        # Save notification preferences
        self.user_config["notify_on_optimization_finish"] = (
            self.notify_on_optimization_finish_checkbox.isChecked()
        )
        self.user_config["notify_on_training_finish"] = (
            self.notify_on_training_finish_checkbox.isChecked()
        )

        Path(self.user_config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_config_path, "w", encoding="utf-8") as f:
            json.dump(self.user_config, f, indent=4)

        if notify:
            StyledMessageBox.information(self, "成功", "用户设置已保存。")

        self.update_user_bar()

    def update_user_bar(self):
        nickname = self.user_config.get("nickname") or "Guest"
        email = self.user_config.get("email") or ""
        display_email = email or "未设置邮箱"
        self.user_nickname_label.setText(nickname)
        self.user_email_label.setText(display_email)

        # Keep user bar text consistent with the global stylesheet.
        self.user_email_label.setStyleSheet("")

    def clear_user_config(self):
        if os.path.exists(self.user_config_path):
            os.remove(self.user_config_path)
        self.load_user_config()  # Reload to get default values
        StyledMessageBox.information(self, "成功", "用户缓存已清除。")

    def create_nav_bar(self):
        self.nav_bar = QFrame()
        self.nav_bar.setObjectName("navBar")
        nav_layout = QVBoxLayout(self.nav_bar)
        nav_layout.setContentsMargins(5, 10, 5, 10)
        nav_layout.setSpacing(15)

        # --- Window Controls (Moved to Top) ---
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(4, 0, 0, 4)
        controls_layout.setSpacing(2)
        self.btn_close = QPushButton("Close")
        self.btn_minimize = QPushButton("Minimize")
        self.btn_maximize = QPushButton("Maximize")

        self.btn_close.setObjectName("btn_close")
        self.btn_minimize.setObjectName("btn_minimize")
        self.btn_maximize.setObjectName("btn_maximize")

        controls_layout.addWidget(self.btn_close)
        controls_layout.addWidget(self.btn_minimize)
        controls_layout.addWidget(self.btn_maximize)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        nav_layout.addLayout(controls_layout)

        self.btn_dashboard = QPushButton(" 训练进程")
        self.btn_settings = QPushButton(" 参数配置")
        self.btn_sentiment = QPushButton(" 情绪数据")
        self.btn_backtest = QPushButton(" 策略回测")
        self.btn_chat = QPushButton(" 智能助手")
        self.btn_logs = QPushButton(" 日志输出")

        buttons = [
            self.btn_dashboard,
            self.btn_settings,
            self.btn_sentiment,
            self.btn_backtest,
            self.btn_chat,
            self.btn_logs,
        ]

        for btn in buttons:
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            nav_layout.addWidget(btn)

        self.btn_dashboard.setChecked(True)

        self.nav_buttons = {
            self.btn_settings: self.settings_page,
            self.btn_sentiment: getattr(self, "sentiment_page", None),
            self.btn_chat: self.chat_page,
            self.btn_logs: self.logs_page,
        }

        # Connect dashboard button to the new state-aware navigation method
        self.btn_dashboard.clicked.connect(self.navigate_to_main_view)

        # Standard page switching for other buttons
        for btn, page in self.nav_buttons.items():
            if page is None:
                continue
            btn.clicked.connect(lambda checked, p=page: self.switch_page(p))

        # Special handling for backtest button
        self.btn_backtest.clicked.connect(self.navigate_to_backtest)

        nav_layout.addStretch()

        # --- User Bar ---
        self.user_bar_button = QPushButton()
        self.user_bar_button.setObjectName("userBarButton")
        user_bar_layout = QVBoxLayout(self.user_bar_button)
        user_bar_layout.setContentsMargins(10, 5, 10, 5)

        self.user_nickname_label = QLabel("Guest")
        font = self.user_nickname_label.font()
        font.setBold(True)
        self.user_nickname_label.setFont(font)

        self.user_email_label = QLabel("Click to configure")
        self.user_email_label.setWordWrap(True)

        user_bar_layout.addWidget(self.user_nickname_label)
        user_bar_layout.addWidget(self.user_email_label)

        nav_layout.addWidget(self.user_bar_button)
        self.user_bar_button.clicked.connect(lambda: self.switch_page(self.user_page))
        self.update_user_bar()  # Initial update

        # Connect window controls
        self.btn_close.clicked.connect(self.close)
        self.btn_minimize.clicked.connect(self.showMinimized)
        self.btn_maximize.clicked.connect(self.toggle_maximize_restore)

    def toggle_maximize_restore(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def navigate_to_main_view(self):
        """Switches to the appropriate main view based on the current training stage."""
        if self.integrated_training_active:
            if self.training_stage == "optimization":
                # If in optimization stage, show the monitor page
                self.switch_page(self.optimization_monitor_page)
            elif self.training_stage == "training":
                # If in final training stage, show the dashboard with the live k-line chart
                self.switch_page(self.dashboard_page)
            else:
                # Fallback to the dashboard if stage is unknown
                self.switch_page(self.dashboard_page)
        else:
            # If no integrated training is active, show the default dashboard
            self.switch_page(self.dashboard_page)

    def navigate_to_backtest(self):
        """Switches to the last viewed backtest page."""
        self.switch_page(self.backtest_current_page)

    def create_main_content(self):
        self.main_content = QFrame()
        self.main_content.setObjectName("mainContent")
        main_layout = QVBoxLayout(self.main_content)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        self.create_dashboard_page()
        self.create_settings_page()
        self.create_sentiment_page()
        self.create_models_page()
        self.create_logs_page()
        self.create_backtest_setup_page()
        self.create_backtest_results_page()
        self.create_optimization_monitor_page()
        self.create_user_page()
        self.create_chat_page()

        self.stacked_widget.addWidget(self.dashboard_page)
        self.stacked_widget.addWidget(self.settings_page)
        self.stacked_widget.addWidget(self.sentiment_page)
        self.stacked_widget.addWidget(self.logs_page)
        self.stacked_widget.addWidget(self.backtest_setup_page)
        self.stacked_widget.addWidget(self.backtest_results_page)
        self.stacked_widget.addWidget(self.optimization_monitor_page)
        self.stacked_widget.addWidget(self.user_page)
        self.stacked_widget.addWidget(self.chat_page)

    def create_user_page(self):
        self.user_page = QWidget()
        main_layout = QVBoxLayout(self.user_page)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("settingsScrollArea")
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        main_card = QFrame()
        main_card.setObjectName("card")
        main_card_layout = QVBoxLayout(main_card)
        main_card_layout.setSpacing(25)

        user_info_group = QGroupBox("基本信息")
        user_info_layout = QFormLayout(user_info_group)
        user_info_layout.setSpacing(15)
        self.nickname_input = QLineEdit()
        self.email_input = QLineEdit()
        user_info_layout.addRow("用户昵称:", self.nickname_input)
        user_info_layout.addRow("用户邮箱:", self.email_input)
        main_card_layout.addWidget(user_info_group)

        notification_group = QGroupBox("通知偏好")
        notification_layout = QVBoxLayout(notification_group)
        notification_layout.setSpacing(10)
        self.notify_on_optimization_finish_checkbox = QCheckBox("优化完成时通知我")
        self.notify_on_training_finish_checkbox = QCheckBox("训练完成时通知我")
        notification_layout.addWidget(self.notify_on_optimization_finish_checkbox)
        notification_layout.addWidget(self.notify_on_training_finish_checkbox)
        main_card_layout.addWidget(notification_group)

        data_group = QGroupBox("数据管理")
        data_layout = QVBoxLayout(data_group)
        data_layout.setSpacing(10)
        self.delete_all_models_button = QPushButton("删除所有模型")
        self.delete_all_models_button.setProperty("cssClass", "dangerButton")
        data_layout.addWidget(self.delete_all_models_button)
        main_card_layout.addWidget(data_group)

        self.save_user_button = QPushButton("保存设置")
        self.save_user_button.setProperty("cssClass", "specialButton")

        layout.addWidget(main_card)
        layout.addWidget(self.save_user_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addStretch()

        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        self.save_user_button.clicked.connect(self.save_user_config)
        self.delete_all_models_button.clicked.connect(self.handle_delete_all_models)

    def create_optimization_monitor_page(self):
        self.optimization_monitor_page = QWidget()
        layout = QVBoxLayout(self.optimization_monitor_page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Top Bar ---
        top_bar_layout = QHBoxLayout()
        self.opt_status_label = QLabel("优化状态: 空闲")
        self.opt_progress_bar = QProgressBar()
        self.stop_opt_button = QPushButton("停止优化")
        top_bar_layout.addWidget(self.opt_status_label)
        top_bar_layout.addWidget(self.opt_progress_bar)
        top_bar_layout.addWidget(self.stop_opt_button)
        layout.addLayout(top_bar_layout)

        # --- Main Content Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Panel: Chart ---
        chart_card = QFrame()
        chart_card.setObjectName("card")
        chart_layout = QVBoxLayout(chart_card)
        chart_layout.addWidget(QLabel("<h3>实时性能</h3>"))
        self.opt_plot_widget = pg.PlotWidget()
        self.opt_plot_widget.setBackground("#282a36")
        self.opt_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.opt_plot_widget.setLabel("left", "得分", color="#CCC")
        self.opt_plot_widget.setLabel("bottom", "试验次数", color="#CCC")
        self.opt_performance_curve = self.opt_plot_widget.plot(
            pen=pg.mkPen("#32AEEF", width=2), symbol="o", symbolBrush="#32AEEF", symbolPen="w"
        )
        self.opt_best_performance_line = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen("#2ECC71", style=Qt.PenStyle.DashLine)
        )
        self.opt_plot_widget.addItem(self.opt_best_performance_line)
        chart_layout.addWidget(self.opt_plot_widget)
        splitter.addWidget(chart_card)

        # --- Right Panel: Details & Logs ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Best Params Card
        best_params_card = QFrame()
        best_params_card.setObjectName("card")
        best_params_layout = QFormLayout(best_params_card)
        best_params_layout.setContentsMargins(15, 15, 15, 15)
        best_params_layout.addRow(QLabel("<h3>当前最佳</h3>"))
        self.opt_best_value_label = QLabel("N/A")
        self.opt_best_params_text = QTextEdit()
        self.opt_best_params_text.setReadOnly(True)
        self.opt_best_params_text.setFixedHeight(150)
        best_params_layout.addRow("最佳得分:", self.opt_best_value_label)
        best_params_layout.addRow(QLabel("最佳参数:"))
        best_params_layout.addRow(self.opt_best_params_text)

        # Log Card
        log_card = QFrame()
        log_card.setObjectName("card")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(15, 15, 15, 15)
        log_layout.addWidget(QLabel("<h3>实时日志</h3>"))
        self.opt_log_display = QTextEdit()
        self.opt_log_display.setReadOnly(True)
        log_layout.addWidget(self.opt_log_display)

        right_layout.addWidget(best_params_card)
        right_layout.addWidget(log_card)
        right_panel.setLayout(right_layout)
        splitter.addWidget(right_panel)

        splitter.setSizes([600, 300])
        layout.addWidget(splitter)

        self.stop_opt_button.clicked.connect(self.stop_process)

    def create_dashboard_page(self):
        self.dashboard_page = QWidget()
        layout = QVBoxLayout(self.dashboard_page)

        top_bar_layout = QHBoxLayout()
        self.integrated_train_button = QPushButton("开始训练")
        self.stop_train_button = QPushButton("停止")
        self.progress_label = QLabel("空闲")
        self.progress_bar = QProgressBar()

        top_bar_layout.addWidget(self.integrated_train_button)
        top_bar_layout.addWidget(self.stop_train_button)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.progress_label)
        top_bar_layout.addWidget(self.progress_bar)

        layout.addLayout(top_bar_layout)

        # 为K线图表添加卡片包装
        kline_card = QFrame()
        kline_card.setObjectName("card")
        kline_card_layout = QVBoxLayout(kline_card)
        kline_card_layout.setContentsMargins(10, 10, 10, 10)
        self.kline_chart = KLineChartWidget()
        kline_card_layout.addWidget(self.kline_chart)
        layout.addWidget(kline_card)

        # --- Connect Buttons ---
        self.integrated_train_button.clicked.connect(self.start_integrated_training)
        self.stop_train_button.clicked.connect(
            self.stop_training_worker
        )  # Connect to the correct stop method

    def _start_in_process_training(self):
        """Starts the final model training in a background thread within the UI process."""
        self.stop_requested = False
        try:
            # Load best_params directly here to ensure they are available
            best_params = {}
            active_best_params_path = find_existing_best_params_path()
            if active_best_params_path and active_best_params_path.exists():
                with active_best_params_path.open("r", encoding="utf-8") as f:
                    best_params = json.load(f)

            best_agent_type = best_params.get("agent_type")
            if best_agent_type:
                combo_index = self.agent_type_combo.findData(best_agent_type)
                if combo_index != -1:
                    self.agent_type_combo.setCurrentIndex(combo_index)

            window_size_value = best_params.get("window_size", self.current_window_size)
            self.current_window_size = window_size_value

            learning_freq_value = best_params.get("n_steps", self.current_learning_frequency)
            self.current_learning_frequency = learning_freq_value

            agent_type_code = (
                self.agent_type_combo.currentData() or best_agent_type or "multiscale_cnn"
            )

            pool_payload = self._resolve_asset_config()
            train_params = {
                "model_name": self.model_name_input.text().strip()
                or f"PPO_Final_Model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "stock_pool": pool_payload["codes"],
                "stock_pool_key": pool_payload["key"],
                "stock_pool_label": pool_payload["label"],
                "start_date": self.start_date_input.date().toString("yyyyMMdd"),
                "end_date": self.end_date_input.date().toString("yyyyMMdd"),
                "commission_rate": float(self.commission_input.text()),
                "slippage": float(self.slippage_input.text()),
                "episodes": int(self.episodes_input.text()),
                "learning_frequency": learning_freq_value,
                "window_size": window_size_value,
                "agent_type": agent_type_code,
            }
            sentiment_dataset = self.sentiment_dataset_combo.currentData()
            if sentiment_dataset:
                train_params["sentiment_dataset"] = sentiment_dataset

            if hasattr(self, "reward_config_editor"):
                reward_text = self.reward_config_editor.toPlainText().strip()
                if reward_text:
                    try:
                        train_params["reward_config"] = json.loads(reward_text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"奖励配置 JSON 无效: {exc}") from exc
        except ValueError as e:
            StyledMessageBox.critical(
                self, "无效输入", f"请确保所有训练参数都是有效数字。\n错误: {e}"
            )
            self._reset_training_state()
            return

        self.kline_chart.clear_all()

        # Prepare k-line background - use cached data if available (integrated training), otherwise fetch fresh data
        try:
            use_cached = (
                self.integrated_training_active
                and self.cached_raw_data is not None
                and not self.cached_raw_data.empty
                and not train_params.get("stock_pool_key")
            )
            if use_cached:
                logging.info("Using cached data from optimization for training.")
                full_kline_data = self.cached_raw_data
            else:
                logging.info(
                    "Fetching fresh data for training (standalone training or stock pool模式)."
                )
                full_kline_data, reference_code = self._fetch_reference_kline(
                    train_params["stock_pool"], train_params["start_date"], train_params["end_date"]
                )
                if full_kline_data is None:
                    raise ValueError("无法为当前股票池获取可用的K线数据。")
                logging.info("K线背景使用标的: %s", reference_code)

            self.kline_chart.plot_background(full_kline_data)
        except Exception as e:
            StyledMessageBox.critical(self, "数据错误", str(e))
            self._reset_training_state()
            return

        self.integrated_train_button.setEnabled(False)
        self.stop_train_button.setEnabled(True)
        self.switch_page(self.dashboard_page)  # Switch to dashboard to see the live chart

        # Use the Worker class to run train() in a separate thread
        # Pass 'self' as dashboard, Worker will provide its own progress_callback
        self.training_worker_obj = Worker(
            train, train_params, dashboard=self, stop_flag_func=lambda: self.stop_requested
        )

        # Connect signals from the worker to the UI slots
        # Note: The worker's progress_callback IS the signals object.
        # We connect our UI update slots to this signals object.
        progress_signals = self.training_worker_obj.signals
        progress_signals.status.connect(self.update_status_label)
        progress_signals.progress.connect(self.update_progress_bar)
        progress_signals.finished.connect(self.on_in_process_training_finished)
        progress_signals.error.connect(self.on_worker_error)

        # The chart_update signal needs to be present on the WorkerSignals object
        # Assuming it is, based on previous code.
        if hasattr(progress_signals, "chart_update"):
            progress_signals.chart_update.connect(self.on_chart_update_live)

        self.threadpool.start(self.training_worker_obj)

    def create_settings_page(self):
        self.settings_page = QWidget()
        main_page_layout = QVBoxLayout(self.settings_page)
        main_page_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("settingsScrollArea")
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        scroll_content_widget = QWidget()
        scroll_content_widget.setObjectName("scrollContent")
        page_layout = QVBoxLayout(scroll_content_widget)
        page_layout.setContentsMargins(20, 20, 20, 20)
        page_layout.setSpacing(20)
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 加载配置以获取默认架构/窗口等参数
        config_data = {}
        try:
            config_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "configs", "hyperparameters.json"
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config_data = {}

        architecture_defaults = config_data.get("model_architecture", {})
        default_agent_type = architecture_defaults.get("default_agent_type", "multiscale_cnn")
        default_window_size = config_data.get("training", {}).get("window_size", 30)
        default_learning_frequency = config_data.get("training", {}).get("update_timestep", 1024)
        self.default_learning_frequency = default_learning_frequency
        self.current_learning_frequency = default_learning_frequency
        self.default_window_size = default_window_size

        train_card = QFrame()
        train_card.setObjectName("card")
        layout = QFormLayout(train_card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        layout.addRow(QLabel("<h3>核心训练参数</h3>"))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name_input = QLineEdit(f"PPO_Model_{timestamp}")
        self.start_date_input = QDateEdit(QDate(2021, 1, 1))
        self.end_date_input = QDateEdit(QDate(2023, 1, 1))
        self.episodes_input = QLineEdit("20")  # Adjusted default for final training
        self.agent_type_combo = QComboBox()
        self.agent_type_combo.addItem("多尺度 CNN 智能体", "multiscale_cnn")
        self.agent_type_combo.addItem("Transformer 智能体", "transformer")
        self.agent_type_combo.addItem("交叉注意力智能体", "cross_attention")
        default_agent_index = self.agent_type_combo.findData(default_agent_type)
        if default_agent_index != -1:
            self.agent_type_combo.setCurrentIndex(default_agent_index)

        self.current_window_size = default_window_size

        layout.addRow(QLabel("新模型名称:"), self.model_name_input)
        layout.addRow(QLabel("开始日期:"), self.start_date_input)
        layout.addRow(QLabel("结束日期:"), self.end_date_input)
        layout.addRow(QLabel("最终训练回合数:"), self.episodes_input)
        layout.addRow(QLabel("智能体架构:"), self.agent_type_combo)

        self.sentiment_dataset_combo = QComboBox()
        self.refresh_sentiment_dataset_options()
        layout.addRow(QLabel("情绪特征数据集:"), self.sentiment_dataset_combo)

        page_layout.addWidget(train_card)

        opt_card = QFrame()
        opt_card.setObjectName("card")
        layout = QFormLayout(opt_card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        layout.addRow(QLabel("<h3>超参数优化参数</h3>"))
        self.opt_trials_input = QLineEdit("100")
        self.opt_timeout_input = QLineEdit("120")  # In minutes
        self.opt_splits_input = QLineEdit("4")
        self.opt_window_min_input = QLineEdit("10")
        self.opt_window_max_input = QLineEdit("60")
        self.opt_multiplier_min_input = QLineEdit("5")
        self.opt_multiplier_max_input = QLineEdit("20")

        layout.addRow(QLabel("Optuna试验次数:"), self.opt_trials_input)
        layout.addRow(QLabel("优化超时时间 (分钟):"), self.opt_timeout_input)
        layout.addRow(QLabel("滚动窗口折叠数:"), self.opt_splits_input)
        layout.addRow(QLabel("窗口大小搜索范围 (Min):"), self.opt_window_min_input)
        layout.addRow(QLabel("窗口大小搜索范围 (Max):"), self.opt_window_max_input)
        layout.addRow(QLabel("训练乘数搜索范围 (Min):"), self.opt_multiplier_min_input)
        layout.addRow(QLabel("训练乘数搜索范围 (Max):"), self.opt_multiplier_max_input)

        self.clear_opt_history_button = QPushButton("清除优化记录")
        layout.addRow(self.clear_opt_history_button)

        page_layout.addWidget(opt_card)

        trade_card = QFrame()
        trade_card.setObjectName("card")
        layout = QFormLayout(trade_card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        layout.addRow(QLabel("<h3>交易参数</h3>"))
        self.commission_input = QLineEdit("0.0003")
        self.slippage_input = QLineEdit("0.0001")

        layout.addRow(QLabel("手续费率:"), self.commission_input)
        layout.addRow(QLabel("滑点:"), self.slippage_input)

        page_layout.addWidget(trade_card)

        reward_card = QFrame()
        reward_card.setObjectName("card")
        reward_layout = QVBoxLayout(reward_card)
        reward_layout.setContentsMargins(20, 20, 20, 20)
        reward_layout.setSpacing(10)
        reward_layout.addWidget(QLabel("<h3>奖励函数配置</h3>"))
        self.reward_config_editor = QTextEdit()
        self.reward_config_editor.setFixedHeight(220)
        default_reward_config = (
            config_data.get("reward") if isinstance(config_data.get("reward"), dict) else {}
        )
        if not default_reward_config:
            try:
                from ..environment.reward import RewardConfig

                default_reward_config = RewardConfig().to_dict()
            except Exception:
                default_reward_config = {}
        self.reward_config_editor.setPlainText(
            json.dumps(default_reward_config, ensure_ascii=False, indent=2)
        )
        reward_layout.addWidget(self.reward_config_editor)
        page_layout.addWidget(reward_card)

        scroll_area.setWidget(scroll_content_widget)
        main_page_layout.addWidget(scroll_area)

    def create_sentiment_page(self):
        self.sentiment_page = QWidget()
        main_layout = QVBoxLayout(self.sentiment_page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        config_card = QFrame()
        config_card.setObjectName("card")
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(20, 20, 20, 20)
        config_layout.setSpacing(12)
        config_layout.addWidget(QLabel("<h3>模型推理设置</h3>"))

        mode_row = QHBoxLayout()
        self.sentiment_local_radio = QRadioButton("本地模型 (Fomalhaut/Ollama)")
        self.sentiment_cloud_radio = QRadioButton("云端模型 (DeepSeek/SiliconFlow)")
        self.sentiment_local_radio.setChecked(True)
        mode_row.addWidget(self.sentiment_local_radio)
        mode_row.addWidget(self.sentiment_cloud_radio)
        mode_row.addStretch()
        config_layout.addLayout(mode_row)

        self.sentiment_mode_stack = QStackedWidget()
        # Local config
        local_widget = QWidget()
        local_form = QFormLayout(local_widget)
        local_form.setSpacing(10)
        self.sentiment_local_host_input = QLineEdit("http://127.0.0.1:11434")
        self.sentiment_local_model_input = QLineEdit("Fomalhaut:latest")
        local_form.addRow("Ollama Host:", self.sentiment_local_host_input)
        local_form.addRow("模型名称:", self.sentiment_local_model_input)
        self.sentiment_mode_stack.addWidget(local_widget)

        # Cloud config
        cloud_widget = QWidget()
        cloud_form = QFormLayout(cloud_widget)
        cloud_form.setSpacing(10)
        self.sentiment_cloud_model_input = QLineEdit(DEFAULT_CLOUD_MODEL)
        self.sentiment_cloud_concurrency_input = QLineEdit("4")
        cloud_form.addRow("模型名称:", self.sentiment_cloud_model_input)
        cloud_form.addRow("并发数:", self.sentiment_cloud_concurrency_input)
        self.sentiment_mode_stack.addWidget(cloud_widget)

        config_layout.addWidget(self.sentiment_mode_stack)

        action_row = QHBoxLayout()
        self.sentiment_run_button = QPushButton("获取新闻并推理情绪")
        self.sentiment_run_button.setProperty("cssClass", "specialButton")
        self.sentiment_refresh_button = QPushButton("刷新情绪数据集")
        action_row.addWidget(self.sentiment_run_button)
        action_row.addWidget(self.sentiment_refresh_button)
        action_row.addStretch()
        config_layout.addLayout(action_row)
        status_row = QHBoxLayout()
        self.sentiment_status_label = QLabel("状态: 空闲")
        self.sentiment_progress_bar = QProgressBar()
        self.sentiment_progress_bar.setRange(0, 100)
        self.sentiment_progress_bar.setValue(0)
        status_row.addWidget(self.sentiment_status_label)
        status_row.addWidget(self.sentiment_progress_bar)
        config_layout.addLayout(status_row)
        main_layout.addWidget(config_card)

        dataset_card = QFrame()
        dataset_card.setObjectName("card")
        dataset_layout = QVBoxLayout(dataset_card)
        dataset_layout.setContentsMargins(20, 20, 20, 20)
        dataset_layout.setSpacing(10)
        dataset_layout.addWidget(QLabel("<h3>情绪数据集</h3>"))
        self.sentiment_dataset_list = QListWidget()
        dataset_layout.addWidget(self.sentiment_dataset_list)
        main_layout.addWidget(dataset_card)

        main_layout.addStretch()

        self.sentiment_local_radio.toggled.connect(self._on_sentiment_mode_changed)
        self.sentiment_refresh_button.clicked.connect(self.refresh_sentiment_dataset_display)
        self.sentiment_run_button.clicked.connect(self.start_sentiment_job)
        self.refresh_sentiment_dataset_display()

    def _on_sentiment_mode_changed(self):
        if self.sentiment_local_radio.isChecked():
            self.sentiment_mode_stack.setCurrentIndex(0)
        else:
            self.sentiment_mode_stack.setCurrentIndex(1)

    def refresh_sentiment_dataset_options(self, select_path: str | None = None):
        combo = getattr(self, "sentiment_dataset_combo", None)
        if combo is None:
            return
        current = select_path or combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("不使用情绪数据", "")
        for dataset_path in list_sentiment_datasets(NEWS_EXPORT_DIR):
            resolved = str(dataset_path.resolve())
            combo.addItem(dataset_path.name, resolved)
        combo.blockSignals(False)
        if current:
            idx = combo.findData(current)
            if idx != -1:
                combo.setCurrentIndex(idx)

    def refresh_sentiment_dataset_display(self):
        widget = getattr(self, "sentiment_dataset_list", None)
        if widget is None:
            return
        widget.clear()
        for dataset_path in list_sentiment_datasets(NEWS_EXPORT_DIR):
            widget.addItem(dataset_path.name)
        self.refresh_sentiment_dataset_options()

    def start_sentiment_job(self):
        if getattr(self, "sentiment_job_running", False):
            StyledMessageBox.information(self, "提示", "情绪任务正在运行，请稍候。")
            return
        start_str = self.start_date_input.date().toString("yyyy-MM-dd")
        end_str = self.end_date_input.date().toString("yyyy-MM-dd")
        label_core = self._sanitize_sentiment_label(
            f"{self.stock_pool_config.get('key', 'pool')}_{self.start_date_input.date().toString('yyyyMMdd')}_{self.end_date_input.date().toString('yyyyMMdd')}_{datetime.datetime.now().strftime('%H%M%S')}"
        )
        csv_path = NEWS_EXPORT_DIR / f"news_{label_core}.csv"
        if self.sentiment_local_radio.isChecked():
            json_path = NEWS_EXPORT_DIR / f"news_fomalhaut_{label_core}.jsonl"
        else:
            json_path = NEWS_EXPORT_DIR / f"news_deepseek_{label_core}.jsonl"
            if not os.environ.get("SILICONFLOW_API_KEY"):
                StyledMessageBox.warning(
                    self,
                    "缺少 API Key",
                    "云端情绪推理需要设置环境变量 SILICONFLOW_API_KEY（建议写入用户环境变量后重启程序）。",
                )
                return

        job_params = {
            "start": start_str,
            "end": end_str,
            "label": label_core,
            "project_root": str(PROJECT_ROOT),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
            "symbols": self._build_sentiment_symbols(),
            "mode": "local" if self.sentiment_local_radio.isChecked() else "cloud",
            "local_host": self.sentiment_local_host_input.text().strip(),
            "local_model": self.sentiment_local_model_input.text().strip(),
            "cloud_concurrency": self.sentiment_cloud_concurrency_input.text().strip() or "4",
            "cloud_url": DEFAULT_CLOUD_URL,
            "cloud_model": self.sentiment_cloud_model_input.text().strip() or DEFAULT_CLOUD_MODEL,
            "cloud_api_key": DEFAULT_CLOUD_API_KEY,
        }
        self.sentiment_status_label.setText("状态: 准备中...")
        if hasattr(self, "sentiment_progress_bar"):
            self.sentiment_progress_bar.setValue(0)
        self.sentiment_run_button.setEnabled(False)
        self.sentiment_job_running = True

        self.sentiment_worker_obj = Worker(self._execute_sentiment_job, job_params)
        signals = self.sentiment_worker_obj.signals
        signals.status.connect(self.append_sentiment_log)
        signals.progress.connect(self.update_sentiment_progress)
        signals.error.connect(self.on_sentiment_job_error)
        signals.finished.connect(self.on_sentiment_job_finished)
        self.threadpool.start(self.sentiment_worker_obj)

    def append_sentiment_log(self, message: str):
        if not message:
            return
        if hasattr(self, "sentiment_status_label"):
            self.sentiment_status_label.setText(f"状态: {message}")

    def on_sentiment_job_error(self, message: str):
        self.sentiment_job_running = False
        self.sentiment_run_button.setEnabled(True)
        status_message = message or "失败"
        self.sentiment_status_label.setText(f"状态: {status_message}")
        if hasattr(self, "sentiment_progress_bar"):
            self.sentiment_progress_bar.setValue(0)

    def on_sentiment_job_finished(self, result):
        self.sentiment_job_running = False
        self.sentiment_run_button.setEnabled(True)
        if not result:
            self.sentiment_status_label.setText("状态: 失败")
            return
        path = result.get("json_path")
        self.sentiment_status_label.setText("状态: 完成")
        if hasattr(self, "sentiment_progress_bar"):
            self.sentiment_progress_bar.setValue(100)
        if path:
            self.refresh_sentiment_dataset_display()
            self.refresh_sentiment_dataset_options(select_path=str(Path(path).resolve()))

    def update_sentiment_progress(self, value: int):
        if hasattr(self, "sentiment_progress_bar"):
            self.sentiment_progress_bar.setValue(value)

    def _execute_sentiment_job(self, job_params: dict, progress_callback=None):
        def emit(msg: str):
            if progress_callback:
                progress_callback.status.emit(msg)

        def set_progress(value: int, status: str | None = None):
            if progress_callback:
                progress_callback.progress.emit(max(0, min(100, value)))
                if status:
                    progress_callback.status.emit(status)

        def run_command(command, label, env=None):
            # Command is constructed by the application and executed without a shell.
            proc = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=job_params["project_root"],
                text=True,
                env=env,
            )
            if proc.stdout is None:
                raise RuntimeError(f"{label} 启动失败：未能打开 stdout 管道")
            for _ in proc.stdout:
                pass
            returncode = proc.wait()
            if returncode != 0:
                raise RuntimeError(f"{label} 执行失败，退出码 {returncode}")

        range_value = f"{job_params['start']}:{job_params['end']}:{job_params['label']}"
        command = [
            sys.executable,
            "-m",
            "news.historical_batches",
            "--range",
            range_value,
            "--per-day-limit",
            "0",
        ]
        symbols = job_params.get("symbols") or []
        if symbols:
            command.extend(["--symbols", *symbols])
        set_progress(5, "抓取新闻...")
        run_command(command, "news.historical_batches")

        csv_path = Path(job_params["csv_path"])
        if not csv_path.exists():
            raise FileNotFoundError(f"未找到导出的新闻文件：{csv_path}")
        emit(f"新闻导出完成，记录保存在 {csv_path}")
        set_progress(40, "新闻抓取完成")

        if job_params["mode"] == "local":
            label_command = [
                sys.executable,
                "-m",
                "news.label_with_fomalhaut",
                "--input",
                str(csv_path),
                "--output",
                job_params["json_path"],
                "--split-output-dir",
                str(NEWS_EXPORT_DIR),
                "--host",
                job_params["local_host"],
                "--model",
                job_params["local_model"],
                "--overwrite",
            ]
            set_progress(60, "本地情绪推理中...")
            run_command(label_command, "label_with_fomalhaut")
        else:
            env = os.environ.copy()
            api_key = (
                job_params.get("cloud_api_key")
                or env.get("SILICONFLOW_API_KEY")
                or DEFAULT_CLOUD_API_KEY
            )
            if api_key:
                env["SILICONFLOW_API_KEY"] = api_key
            label_command = [
                sys.executable,
                "-m",
                "news.label_with_deepseek",
                "--input",
                str(csv_path),
                "--output",
                job_params["json_path"],
                "--split-output-dir",
                str(NEWS_EXPORT_DIR),
                "--api-url",
                job_params["cloud_url"],
                "--model",
                job_params["cloud_model"],
                "--concurrency",
                job_params["cloud_concurrency"],
                "--overwrite",
            ]
            set_progress(60, "云端情绪推理中...")
            run_command(label_command, "label_with_deepseek", env=env)

        json_path = Path(job_params["json_path"])
        if not json_path.exists():
            raise FileNotFoundError(f"情绪结果未生成：{json_path}")
        emit(f"情绪推理已完成：{json_path}")
        set_progress(100, "完成")
        return {"json_path": str(json_path)}

    def _build_sentiment_symbols(self):
        symbols = []
        for code in self.stock_pool_config.get("codes", []):
            token = code.upper()
            if token.endswith(".SH"):
                symbols.append(f"SH{token[:-3]}")
            elif token.endswith(".SZ"):
                symbols.append(f"SZ{token[:-3]}")
            else:
                symbols.append(token)
        return symbols

    def _sanitize_sentiment_label(self, value: str) -> str:
        safe = []
        for ch in value:
            if ch.isalnum() or ch in ("_", "-"):
                safe.append(ch)
            else:
                safe.append("_")
        return "".join(safe)

    def create_models_page(self):
        self.models_page = QWidget()
        layout = QHBoxLayout(self.models_page)
        layout.setContentsMargins(20, 20, 20, 20)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Panel: Model List ---
        list_card = QFrame()
        list_card.setObjectName("card")
        list_layout = QVBoxLayout(list_card)
        list_layout.addWidget(QLabel("<h3>已存模型</h3>"))
        self.model_list_widget = QListWidget()
        list_layout.addWidget(self.model_list_widget)

        # --- Right Panel: Details and Actions ---
        details_card = QFrame()
        details_card.setObjectName("card")
        details_layout = QVBoxLayout(details_card)
        details_layout.setSpacing(15)
        details_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        details_layout.addWidget(QLabel("<h3>模型详情与操作</h3>"))

        self.refresh_models_button = QPushButton("刷新列表")
        self.view_history_button = QPushButton("查看训练记录")
        self.delete_model_button = QPushButton("删除选中模型")

        actions_layout = QHBoxLayout()
        actions_layout.addWidget(self.refresh_models_button)
        actions_layout.addWidget(self.view_history_button)
        actions_layout.addWidget(self.delete_model_button)
        actions_layout.addStretch()
        details_layout.addLayout(actions_layout)

        # History Section
        history_group = QFrame()  # Using a frame for visual grouping
        history_group_layout = QVBoxLayout(history_group)
        history_group_layout.setContentsMargins(0, 10, 0, 0)

        history_group_layout.addWidget(QLabel("<h4>训练会话</h4>"))
        self.session_list_widget = QListWidget()
        self.session_list_widget.setMaximumHeight(200)
        history_group_layout.addWidget(self.session_list_widget)

        history_group_layout.addWidget(QLabel("<h4>选择回合</h4>"))
        self.episode_combo_box = QComboBox()
        history_group_layout.addWidget(self.episode_combo_box)

        details_layout.addWidget(history_group)
        details_layout.addStretch()

        splitter.addWidget(list_card)
        splitter.addWidget(details_card)
        splitter.setSizes([300, 500])  # Initial size distribution

        layout.addWidget(splitter)

    def create_logs_page(self):
        self.logs_page = QWidget()
        layout = QVBoxLayout(self.logs_page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        self.clear_logs_button = QPushButton("清除日志")

        toolbar_layout.addWidget(QLabel("日志级别:"))
        toolbar_layout.addWidget(self.log_level_combo)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(self.clear_logs_button)

        layout.addLayout(toolbar_layout)

        # --- Log Display ---
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)

        # --- Connections ---
        self.clear_logs_button.clicked.connect(self.log_display.clear)
        # Note: Filtering logic would need to be connected to the logger handler.
        # This is a placeholder for now.
        # self.log_level_combo.currentTextChanged.connect(self.on_log_level_changed) # Logic moved to main.py

    def create_backtest_setup_page(self):
        self.backtest_setup_page = QWidget()
        page_layout = QVBoxLayout(self.backtest_setup_page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        page_layout.setSpacing(20)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        page_layout.addWidget(splitter)

        # --- Left Panel: Model Management ---
        model_panel = QFrame()
        model_panel.setObjectName("card")
        model_panel_layout = QVBoxLayout(model_panel)
        model_panel_layout.setContentsMargins(18, 18, 18, 18)
        model_panel_layout.setSpacing(12)

        header_layout = QHBoxLayout()
        title_label = QLabel("选择回测模型")
        title_label.setProperty("cssClass", "title")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        self.backtest_refresh_button = QPushButton("刷新")
        header_layout.addWidget(self.backtest_refresh_button)
        model_panel_layout.addLayout(header_layout)

        self.backtest_search_input = QLineEdit()
        self.backtest_search_input.setPlaceholderText("搜索模型名称…")
        model_panel_layout.addWidget(self.backtest_search_input)

        self.backtest_model_list = QListWidget()
        self.backtest_model_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        model_panel_layout.addWidget(self.backtest_model_list, stretch=1)

        action_row = QHBoxLayout()
        self.backtest_delete_model_button = QPushButton("删除选中模型")
        self.backtest_delete_model_button.setProperty("cssClass", "dangerButton")
        action_row.addWidget(self.backtest_delete_model_button)
        action_row.addStretch()
        model_panel_layout.addLayout(action_row)

        splitter.addWidget(model_panel)

        # --- Right Panel: Parameters ---
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(15)

        params_card = QFrame()
        params_card.setObjectName("card")
        params_layout = QVBoxLayout(params_card)
        params_layout.setContentsMargins(18, 18, 18, 18)
        params_layout.setSpacing(15)
        params_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        params_title = QLabel("回测参数")
        params_title.setProperty("cssClass", "title")
        params_layout.addWidget(params_title)

        params_form_layout = QFormLayout()
        params_form_layout.setSpacing(12)
        self.backtest_start_date = QDateEdit(QDate(2023, 1, 1))
        self.backtest_start_date.setCalendarPopup(True)
        self.backtest_end_date = QDateEdit(QDate.currentDate())
        self.backtest_end_date.setCalendarPopup(True)
        params_form_layout.addRow("开始日期:", self.backtest_start_date)
        params_form_layout.addRow("结束日期:", self.backtest_end_date)
        params_layout.addLayout(params_form_layout)
        params_layout.addStretch(1)
        self.start_backtest_button = QPushButton("开始回测")
        self.start_backtest_button.setProperty("cssClass", "specialButton")
        self.start_backtest_button.setMinimumHeight(36)
        params_layout.addWidget(self.start_backtest_button, 0, Qt.AlignmentFlag.AlignRight)

        config_layout.addWidget(params_card, 1)

        splitter.addWidget(config_panel)
        splitter.setSizes([600, 320])

        self.backtest_delete_model_button.clicked.connect(self.handle_delete_selected_model)
        self.backtest_refresh_button.clicked.connect(self.refresh_backtest_model_list)
        self.backtest_search_input.textChanged.connect(self.apply_backtest_model_filter)
        self._backtest_models = []
        self.refresh_backtest_model_list()

    def refresh_backtest_model_list(self):
        list_widget = getattr(self, "backtest_model_list", None)
        if list_widget is None:
            logging.debug("Backtest model list widget unavailable; skipping refresh.")
            return
        model_paths = list_model_zip_paths()
        self._backtest_models = [str(path) for path in model_paths]
        logging.debug("Backtest model scan finished: found %d models.", len(self._backtest_models))
        self.apply_backtest_model_filter()

    def apply_backtest_model_filter(self, *_args):
        list_widget = getattr(self, "backtest_model_list", None)
        if list_widget is None:
            return
        delete_button = getattr(self, "backtest_delete_model_button", None)
        search_box = getattr(self, "backtest_search_input", None)
        query = ""
        if search_box:
            query = search_box.text().strip().lower()

        current_path = None
        current_item = list_widget.currentItem()
        if current_item:
            current_path = current_item.data(Qt.ItemDataRole.UserRole)

        list_widget.blockSignals(True)
        list_widget.clear()
        for path in getattr(self, "_backtest_models", []):
            name = os.path.basename(path)
            if query and query not in name.lower():
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            list_widget.addItem(item)
        list_widget.blockSignals(False)

        has_items = list_widget.count() > 0
        if delete_button:
            delete_button.setEnabled(has_items)
        if has_items:
            target_row = 0
            if current_path:
                for idx in range(list_widget.count()):
                    if list_widget.item(idx).data(Qt.ItemDataRole.UserRole) == current_path:
                        target_row = idx
                        break
            list_widget.setCurrentRow(target_row)
        else:
            list_widget.setCurrentRow(-1)

    def handle_delete_selected_model(self):
        list_widget = getattr(self, "backtest_model_list", None)
        if not list_widget:
            StyledMessageBox.warning(self, "提示", "当前界面不支持模型删除。")
            return
        current_item = list_widget.currentItem()
        if not current_item:
            StyledMessageBox.information(self, "提示", "请先选择要删除的模型。")
            return
        model_path = current_item.data(Qt.ItemDataRole.UserRole)
        if not model_path:
            StyledMessageBox.information(self, "提示", "请选择有效的模型文件。")
            return
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            StyledMessageBox.information(self, "提示", "模型文件不存在，已刷新列表。")
            self.refresh_backtest_model_list()
            return
        if not is_user_model_path(model_path_obj):
            StyledMessageBox.warning(
                self,
                "提示",
                "该模型为仓库内置/只读模型，无法通过界面删除。",
            )
            return
        model_name = model_path_obj.name
        if not StyledMessageBox.question(
            self, "确认操作", f"确定要删除模型 {model_name} 吗？\n此操作无法撤销。"
        ):
            return
        try:
            model_path_obj.unlink()
        except Exception as exc:
            StyledMessageBox.critical(self, "错误", f"删除模型失败：{exc}")
            return
        StyledMessageBox.information(self, "成功", f"已删除模型 {model_name}。")
        self.refresh_backtest_model_list()

    def create_backtest_results_page(self):
        self.backtest_results_page = QWidget()
        layout = QVBoxLayout(self.backtest_results_page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        status_card = QFrame()
        status_card.setObjectName("card")
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(15, 12, 15, 12)
        status_layout.setSpacing(10)
        self.backtest_status_label = QLabel("准备回测...")
        self.backtest_status_label.setProperty("cssClass", "title")
        status_layout.addWidget(self.backtest_status_label)
        status_layout.addStretch()
        self.toggle_metrics_button = QPushButton("显示指标")
        self.toggle_metrics_button.setCheckable(True)
        self.back_to_setup_button = QPushButton("返回配置")
        status_layout.addWidget(self.toggle_metrics_button)
        status_layout.addWidget(self.back_to_setup_button)
        layout.addWidget(status_card)

        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)
        layout.addWidget(content_splitter, 1)

        backtest_kline_card = QFrame()
        backtest_kline_card.setObjectName("card")
        backtest_kline_layout = QVBoxLayout(backtest_kline_card)
        backtest_kline_layout.setContentsMargins(12, 12, 12, 12)
        backtest_kline_layout.setSpacing(10)
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        header_row.addWidget(QLabel("股票选择:"))
        self.stock_selector = QComboBox()
        self.stock_selector.addItem("自动选择", None)
        self.stock_selector.currentIndexChanged.connect(self._on_stock_selection_changed)
        header_row.addWidget(self.stock_selector, stretch=1)
        self.summary_label = QLabel("尚未回测")
        self.summary_label.setObjectName("summaryLabel")
        header_row.addWidget(self.summary_label, alignment=Qt.AlignmentFlag.AlignRight)
        backtest_kline_layout.addLayout(header_row)

        self.backtest_kline_chart = KLineChartWidget()
        backtest_kline_layout.addWidget(self.backtest_kline_chart)
        content_splitter.addWidget(backtest_kline_card)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_container = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        self.sidebar_stack = QStackedWidget()

        self.metrics_panel = QFrame()
        self.metrics_panel.setObjectName("card")
        metrics_layout = QVBoxLayout(self.metrics_panel)
        metrics_layout.setContentsMargins(15, 15, 15, 15)
        metrics_layout.setSpacing(10)
        metrics_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        metrics_title = QLabel("回测性能指标")
        metrics_title.setProperty("cssClass", "title")
        metrics_layout.addWidget(metrics_title)
        self.metrics_scroll_area = QScrollArea()
        self.metrics_scroll_area.setWidgetResizable(True)
        self.metrics_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        metrics_layout.addWidget(self.metrics_scroll_area)
        self.metrics_container = QWidget()
        self.metrics_display_layout = QFormLayout(self.metrics_container)
        self.metrics_display_layout.setSpacing(10)
        self.metrics_scroll_area.setWidget(self.metrics_container)
        self.sidebar_stack.addWidget(self.metrics_panel)

        self.trade_panel = QFrame()
        self.trade_panel.setObjectName("card")
        trade_layout = QVBoxLayout(self.trade_panel)
        trade_layout.setContentsMargins(15, 15, 15, 15)
        trade_layout.setSpacing(8)
        trade_title = QLabel("交易明细")
        trade_title.setProperty("cssClass", "title")
        trade_layout.addWidget(trade_title)
        self.trade_details_table = QTableWidget()
        self.trade_details_table.setColumnCount(5)
        self.trade_details_table.setHorizontalHeaderLabels(["时间", "方向", "价格", "数量", "标的"])
        header = self.trade_details_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_details_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.trade_details_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        trade_layout.addWidget(self.trade_details_table, 1)
        self.sidebar_stack.addWidget(self.trade_panel)

        sidebar_layout.addWidget(self.sidebar_stack, 1)

        sidebar_scroll.setWidget(sidebar_container)
        content_splitter.addWidget(sidebar_scroll)
        content_splitter.setSizes([700, 320])

        self.sidebar_stack.setCurrentWidget(self.trade_panel)
        self.toggle_metrics_button.toggled.connect(self.toggle_metrics_panel)
        self.toggle_metrics_panel(False)

    def _on_stock_selection_changed(self, index):
        if not getattr(self, "stock_selector", None) or sip.isdeleted(self.stock_selector):
            return
        symbol = self.stock_selector.itemData(index)
        self._update_backtest_chart(symbol)

    def _format_date_range(self, start, end):
        def fmt(val):
            if not val:
                return ""
            if len(val) == 8:
                return f"{val[0:4]}-{val[4:6]}-{val[6:]}"
            return val

        start_fmt = fmt(start)
        end_fmt = fmt(end)
        if start_fmt and end_fmt:
            return f"{start_fmt} 至 {end_fmt}"
        return start_fmt or end_fmt or ""

    def _update_backtest_chart(self, symbol):
        if not getattr(self, "backtest_kline_chart", None):
            return
        results = self.current_backtest_results
        if results is None:
            return
        if (
            symbol is None
            and getattr(self, "stock_selector", None)
            and not sip.isdeleted(self.stock_selector)
        ):
            symbol = self.stock_selector.currentData()
        base_kline = results.get("base_kline")
        base_symbol = results.get("base_symbol") or results.get("stock_code")
        symbol_kline_map = results.get("symbol_kline_map") or {}
        symbol_trades_map = results.get("symbol_trades_map") or {}
        portfolio_history = results.get("portfolio_history")
        stock_pool = results.get("stock_pool") or []
        is_pool_mode = len(stock_pool) > 1
        if symbol:
            kline = symbol_kline_map.get(symbol)
            trades = symbol_trades_map.get(symbol)
            label = symbol
        else:
            kline = base_kline
            trades = symbol_trades_map.get(base_symbol)
            if trades is None:
                trades = results.get("trades")
            if is_pool_mode:
                trades = None
            label = base_symbol or "上证50ETF"
        if kline is not None and not kline.empty:
            self.backtest_kline_chart.plot_background(kline)
        else:
            self.backtest_kline_chart.clear_background()
        self.backtest_kline_chart.update_overlays(trades, portfolio_history)
        date_range = self._format_date_range(results.get("start_date"), results.get("end_date"))
        summary_text = f"{label}"
        if date_range:
            summary_text += f" · {date_range}"
        self.summary_label.setText(summary_text)
        target_symbol = symbol or base_symbol
        self._update_trade_details_table(target_symbol)

    def _update_trade_details_table(self, symbol):
        table = getattr(self, "trade_details_table", None)
        if not table:
            return
        table.setRowCount(0)
        results = getattr(self, "current_backtest_results", None)
        if not results:
            return
        trades_df = None
        symbol_map = results.get("symbol_trades_map") or {}
        if symbol:
            trades_df = symbol_map.get(symbol)
        if trades_df is None or getattr(trades_df, "empty", True):
            trades_df = results.get("trades")
            if trades_df is None or getattr(trades_df, "empty", True):
                return
            if symbol and "symbol" in trades_df.columns:
                trades_df = trades_df[trades_df["symbol"] == symbol]
        if trades_df is None or getattr(trades_df, "empty", True):
            return
        try:
            rows_df = trades_df.reset_index()
        except Exception:
            rows_df = pd.DataFrame(trades_df)
        rows = len(rows_df)
        table.setRowCount(rows)
        table.verticalHeader().setVisible(False)
        for row in range(rows):
            record = rows_df.iloc[row]
            timestamp = record.get("timestamp")
            if hasattr(timestamp, "strftime"):
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
            else:
                timestamp_str = str(timestamp)
            side = record.get("side", "-")
            if isinstance(side, str):
                upper = side.upper()
                if upper == "BUY":
                    side = "买入"
                elif upper == "SELL":
                    side = "卖出"
            price = record.get("price", "")
            quantity = record.get("quantity", "")
            symbol_value = record.get("symbol", symbol or "-")
            values = [
                timestamp_str,
                str(side),
                f"{float(price):.4f}" if isinstance(price, (int, float)) else str(price),
                f"{float(quantity):.4f}" if isinstance(quantity, (int, float)) else str(quantity),
                str(symbol_value),
            ]
            for col, value in enumerate(values):
                table.setItem(row, col, QTableWidgetItem(value))

    def toggle_metrics_panel(self, checked):
        sidebar_stack = getattr(self, "sidebar_stack", None)
        trade_panel = getattr(self, "trade_panel", None)
        metrics_panel = getattr(self, "metrics_panel", None)
        if sidebar_stack and metrics_panel and trade_panel:
            target = metrics_panel if checked else trade_panel
            sidebar_stack.setCurrentWidget(target)
        self.toggle_metrics_button.setText("隐藏指标" if checked else "显示指标")

    def _current_selection(self):
        return self.stock_pool_config

    def _is_pool_mode(self):
        return True

    def _get_optimization_target(self):
        """
        当前固定返回上证50股票池配置，用于超参数优化。
        """
        return self.stock_pool_config

    def _resolve_asset_config(self):
        return self.stock_pool_config

    def _fetch_reference_kline(self, codes, start_date, end_date):
        for code in codes or []:
            data = get_stock_data(code, start_date, end_date)
            if data is not None and not data.empty:
                return data, code
        return None, None

    def display_backtest_results(self, results):
        """Helper function to update the results page with simulation data."""
        if not getattr(self, "stock_selector", None) or sip.isdeleted(self.stock_selector):
            return
        if not getattr(self, "backtest_kline_chart", None):
            return

        results = results if results is not None else {}
        self.current_backtest_results = results

        # Populate stock selector（默认展示上证50ETF基准）
        self.stock_selector.blockSignals(True)
        self.stock_selector.clear()
        base_symbol = results.get("base_symbol") or results.get("stock_code")
        base_label = f"{base_symbol}（上证50ETF）" if base_symbol else "上证50ETF"
        self.stock_selector.addItem(base_label, None)
        symbol_kline_map = results.get("symbol_kline_map") or {}
        for symbol in sorted(symbol_kline_map.keys()):
            if symbol == base_symbol:
                continue
            self.stock_selector.addItem(symbol, symbol)
        self.stock_selector.blockSignals(False)
        self.stock_selector.setCurrentIndex(0)
        self._update_backtest_chart(None)

        # Calculate and display metrics
        benchmark_df = results.get("base_kline")
        if benchmark_df is None or getattr(benchmark_df, "empty", False):
            kline_data = results.get("kline_data")
            if kline_data is not None and not kline_data.empty:
                benchmark_df = kline_data
        benchmark_series = results.get("benchmark_series")
        if (
            benchmark_df is None or getattr(benchmark_df, "empty", False)
        ) and benchmark_series is not None:
            benchmark_df = pd.DataFrame(
                {
                    "open": benchmark_series,
                    "high": benchmark_series,
                    "low": benchmark_series,
                    "close": benchmark_series,
                    "volume": 0,
                }
            )

        portfolio_history = results.get("portfolio_history")
        trades_df = results.get("trades")
        if trades_df is None:
            trades_df = pd.DataFrame()
        extra_metrics = {}
        if trades_df is not None and not trades_df.empty:
            total_trades = len(trades_df)
            extra_metrics["交易笔数"] = total_trades
            if "symbol" in trades_df.columns:
                extra_metrics["交易标的数"] = trades_df["symbol"].nunique()
            if "side" in trades_df.columns:
                sides = trades_df["side"].astype(str).str.upper()
                extra_metrics["买入次数"] = int((sides == "BUY").sum())
                extra_metrics["卖出次数"] = int((sides == "SELL").sum())
            if "quantity" in trades_df.columns:
                total_qty = float(trades_df["quantity"].sum())
                extra_metrics["总成交数量"] = f"{total_qty:.2f}"
                if total_trades:
                    extra_metrics["平均成交数量"] = f"{(total_qty / total_trades):.2f}"
            if {"price", "quantity"}.issubset(trades_df.columns):
                traded_value = float((trades_df["price"] * trades_df["quantity"]).sum())
                extra_metrics["成交额"] = f"{traded_value:.2f}"
            elif "price" in trades_df.columns:
                extra_metrics["平均成交价格"] = f"{float(trades_df['price'].mean()):.4f}"
            if len(trades_df.index):
                first_ts = trades_df.index.min()
                last_ts = trades_df.index.max()
                start_str = (
                    first_ts.strftime("%Y%m%d") if hasattr(first_ts, "strftime") else str(first_ts)
                )
                end_str = (
                    last_ts.strftime("%Y%m%d") if hasattr(last_ts, "strftime") else str(last_ts)
                )
                date_span = self._format_date_range(start_str, end_str)
                if date_span:
                    extra_metrics["交易时间范围"] = date_span
        if portfolio_history is not None:
            try:
                metrics = calculate_performance_metrics(
                    portfolio_history, trades_df, kline_data=benchmark_df
                )
            except Exception as exc:
                logger.exception("[Backtest] 指标计算失败: %s", exc)
                metrics = {"指标状态": "计算失败"}
        else:
            metrics = {"指标状态": "暂无数据"}

        while self.metrics_display_layout.count():
            item = self.metrics_display_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        metric_items = list(metrics.items())
        metric_items.extend(extra_metrics.items())
        for name, value in metric_items:
            self.metrics_display_layout.addRow(QLabel(f"{name}:"), QLabel(str(value)))

        self.backtest_status_label.setText("回测完成")
        self.switch_page(self.backtest_results_page)
        self.backtest_current_page = self.backtest_results_page

    def get_train_parameters(self):
        model_name = self.model_name_input.text().strip()
        if not model_name:
            model_name = f"PPO_Model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        window_size_value = getattr(self, "current_window_size", self.default_window_size)
        pool_payload = self._resolve_asset_config()

        return {
            "model_name": model_name,
            "stock_pool": pool_payload["codes"],
            "stock_pool_key": pool_payload["key"],
            "stock_pool_label": pool_payload["label"],
            "start_date": self.start_date_input.date().toString("yyyyMMdd"),
            "end_date": self.end_date_input.date().toString("yyyyMMdd"),
            "commission_rate": float(self.commission_input.text()),
            "slippage": float(self.slippage_input.text()),
            "episodes": int(self.episodes_input.text()),
            "learning_frequency": self.current_learning_frequency,
            "window_size": window_size_value,
            "agent_type": self.agent_type_combo.currentData(),
        }

    def start_optimization(self):
        if self._is_pool_mode():
            StyledMessageBox.information(
                self,
                "提示",
                "当前仅提供上证50股票池，将使用默认成分股中的代表性标的执行超参数搜索。",
            )
        # Reset UI elements
        self.opt_performance_curve.setData([], [])
        self.opt_best_performance_line.setPos(0)
        self.opt_best_value_label.setText("N/A")
        self.opt_best_params_text.clear()
        self.opt_log_display.clear()

        # Switch to the monitor page
        self.switch_page(self.optimization_monitor_page)

        opt_target = self._get_optimization_target()
        stock_code = opt_target["codes"][0]
        command = [
            sys.executable,
            "-m",
            "virgo_trader.training.optimize",
            "--stock_code",
            stock_code,
            "--n_trials",
            self.opt_trials_input.text(),
            "--timeout",
            self.opt_timeout_input.text(),
            "--start_date",
            self.start_date_input.date().toString("yyyyMMdd"),
            "--end_date",
            self.end_date_input.date().toString("yyyyMMdd"),
            "--n_splits",
            self.opt_splits_input.text(),
            "--window_size_min",
            self.opt_window_min_input.text(),
            "--window_size_max",
            self.opt_window_max_input.text(),
            "--multiplier_min",
            self.opt_multiplier_min_input.text(),
            "--multiplier_max",
            self.opt_multiplier_max_input.text(),
        ]
        self.current_optimization_code = stock_code
        self.run_external_process(command, is_optimization=True)

    def start_integrated_training(self):
        """
        启动集成训练流程.
        如果 best_params.json 存在, 直接使用它训练最终模型.
        否则, 先运行优化, 然后再训练.
        """
        if self.worker_thread and self.worker_thread.isRunning():
            StyledMessageBox.warning(self, "Process Running", "训练进程正在运行中，请稍后再试。")
            return

        self.integrated_training_active = True
        # Clear data cache at the start of integrated training
        self.cached_data = None
        self.cached_raw_data = None

        if find_existing_best_params_path():
            # --- Directly start final training IN-PROCESS ---
            self.training_stage = "training"
            self.progress_label.setText(
                "Stage 1/1: Final Model Training (using existing params)..."
            )
            self._start_in_process_training()

        else:
            # --- 启动完整的优化 -> 训练流程 ---
            self.training_stage = "optimization"

            # 重置优化相关UI
            self.opt_performance_curve.setData([], [])
            self.opt_best_performance_line.setPos(0)
            self.opt_best_value_label.setText("N/A")
            self.opt_best_params_text.clear()
            self.opt_log_display.clear()

            # 切换到优化监控页面
            self.switch_page(self.optimization_monitor_page)

            # 更新状态标签
            self.opt_status_label.setText("第一阶段 - 参数优化中...")
            self.progress_label.setText("Stage 1/2: Parameter Optimization...")

            # 启动优化
            try:
                opt_target = self._get_optimization_target()
                self.current_optimization_code = opt_target["codes"][0]
                command = [
                    sys.executable,
                    "-m",
                    "virgo_trader.training.optimize",
                    "--n_trials",
                    self.opt_trials_input.text(),
                    "--timeout",
                    self.opt_timeout_input.text(),
                    "--stock_code",
                    self.current_optimization_code,
                    "--start_date",
                    self.start_date_input.date().toString("yyyyMMdd"),
                    "--end_date",
                    self.end_date_input.date().toString("yyyyMMdd"),
                    "--n_splits",
                    self.opt_splits_input.text(),
                    "--window_size_min",
                    self.opt_window_min_input.text(),
                    "--window_size_max",
                    self.opt_window_max_input.text(),
                    "--multiplier_min",
                    self.opt_multiplier_min_input.text(),
                    "--multiplier_max",
                    self.opt_multiplier_max_input.text(),
                ]
                self.run_external_process(command, is_optimization=True, is_integrated=True)
            except ValueError:
                StyledMessageBox.critical(
                    self,
                    "Invalid Input",
                    "Please ensure all optimization parameters are valid numbers.",
                )
                self._reset_training_state()

    def start_final_train(self):
        # This method is now obsolete due to the integrated flow, but can be kept for potential future use.
        try:
            pool_payload = self._resolve_asset_config()
            command = [
                sys.executable,
                "-m",
                "virgo_trader.train",
                "--start_date",
                self.start_date_input.date().toString("yyyyMMdd"),
                "--end_date",
                self.end_date_input.date().toString("yyyyMMdd"),
                "--episodes",
                self.episodes_input.text(),
                "--commission_rate",
                self.commission_input.text(),
                "--slippage",
                self.slippage_input.text(),
            ]
            command.extend(["--stock_pool", pool_payload["key"]])
            self.run_external_process(command)
        except ValueError as exc:
            StyledMessageBox.critical(self, "Invalid Input", f"参数错误: {exc}")

    def run_external_process(self, command, is_optimization=False, is_integrated=False):
        if self.worker_thread and self.worker_thread.isRunning():
            StyledMessageBox.warning(self, "Process Running", "Another process is already running.")
            return

        self.worker = ExternalProcessWorker(command)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        self.worker.output_received.connect(self.append_log)
        self.worker.process_finished.connect(self.on_process_finished)
        self.worker_thread.started.connect(self.worker.run)

        self.worker_thread.start()

        if is_optimization:
            self.worker.output_received.connect(self.append_optimization_log)
            if is_integrated:
                # 集成训练的优化阶段
                self.opt_status_label.setText("第一阶段 - 参数优化中...")
                self.progress_label.setText("Stage 1/2: Parameter Optimization...")
            else:
                # 单独的优化
                self.opt_status_label.setText("优化状态: 运行中...")
                self.progress_label.setText("Running Optimization...")
            self.optimization_timer.start(1000)  # Check for updates every second
        else:
            self.progress_label.setText("Running...")
            if not self.integrated_training_active:
                self.switch_page(self.logs_page)

    def update_optimization_monitor(self):
        try:
            progress_path = find_existing_progress_file()
            if not progress_path:
                return

            with progress_path.open("r", encoding="utf-8") as f:
                progress_data = json.load(f)

            trials = progress_data.get("trials", [])
            if not trials:
                return

            trial_numbers = [t["number"] for t in trials]
            values = [t["value"] for t in trials]

            self.opt_performance_curve.setData(trial_numbers, values)

            best_trial = progress_data.get("best_trial")
            if best_trial:
                self.opt_best_value_label.setText(f"{best_trial['value']:.4f}")

                new_params_text = json.dumps(best_trial["params"], indent=4)
                if self.opt_best_params_text.toPlainText() != new_params_text:
                    self.opt_best_params_text.setText(new_params_text)

                self.opt_best_performance_line.setPos(best_trial["value"])

            total_trials = int(self.opt_trials_input.text())
            self.opt_progress_bar.setMaximum(total_trials)
            self.opt_progress_bar.setValue(len(trials))

        except (FileNotFoundError, json.JSONDecodeError):
            # File might not be created yet or is being written to
            pass

    def stop_process(self):
        """Stops the EXTERNAL process (like optimization)."""
        if self.worker:
            self.worker.stop()
            self.progress_label.setText("Stopping Optimization...")

        if self.integrated_training_active:
            self._reset_training_state()
            self.opt_status_label.setText("Optimization Stopped.")

    def stop_training_worker(self):
        """Stops BOTH external and in-process workers."""
        # Check for and stop the external process worker
        if self.worker and self.worker_thread and self.worker_thread.isRunning():
            self.stop_process()

        # Check for and stop the in-process training worker
        if self.training_worker_obj:
            import logging

            logging.info("Stop training requested by user.")
            self.stop_requested = True
            self.stop_train_button.setEnabled(False)
            self.progress_label.setText("Stopping Training...")

    def append_log(self, text):
        self.log_display.append(text.strip())

    def append_optimization_log(self, text):
        self.opt_log_display.append(text.strip())

    def on_process_finished(self, return_code):
        # 处理集成训练的两阶段流程
        if self.integrated_training_active:
            if self.training_stage == "optimization":
                # 优化阶段完成
                if self.optimization_timer.isActive():
                    self.optimization_timer.stop()
                    self.update_optimization_monitor()

                if return_code == 0:
                    # 优化成功，缓存数据以备训练使用
                    try:
                        stock_code = (
                            self.current_optimization_code or self.stock_pool_config["codes"][0]
                        )
                        logging.info(
                            "股票池模式不缓存单一标的数据，训练阶段将重新拉取组合数据（参考标的 %s）。",
                            stock_code,
                        )
                        self.cached_raw_data = None
                    except Exception as e:
                        logging.warning(
                            f"Failed to prepare cached data: {e}. Training will fetch data normally."
                        )
                        self.cached_raw_data = None

                    # 优化成功，开始第二阶段：训练最终模型
                    self.training_stage = "training"
                    self.opt_status_label.setText("第二阶段 - 最终模型训练中...")
                    self.progress_label.setText("Stage 2/2: Final Model Training...")

                    # 清理当前worker
                    if self.worker_thread:
                        self.worker_thread.quit()
                        self.worker_thread.wait()
                        self.worker_thread = None
                        self.worker = None

                    # Start the final training IN-PROCESS
                    self._start_in_process_training()
                    return
                else:
                    # 优化失败
                    self.opt_status_label.setText(f"训练失败: 优化阶段返回码 {return_code}")
                    self.progress_label.setText(
                        f"Integration Training Failed: Optimization code {return_code}"
                    )
                    self._reset_training_state()

            elif self.training_stage == "training":
                # 训练阶段完成
                if return_code == 0:
                    self.opt_status_label.setText("训练完成: 优化和训练都已成功!")
                    self.progress_label.setText("Integration Training Completed Successfully!")
                else:
                    self.opt_status_label.setText(f"训练失败: 训练阶段返回码 {return_code}")
                    self.progress_label.setText(
                        f"Integration Training Failed: Training code {return_code}"
                    )
                self._reset_training_state()
        else:
            # 非集成训练模式的处理
            self.progress_label.setText(f"Finished with code: {return_code}")

            if self.optimization_timer.isActive():
                self.optimization_timer.stop()
                self.opt_status_label.setText(f"优化完成，返回码: {return_code}")
                # Final update
                self.update_optimization_monitor()

        # 清理worker线程
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None

    # --- New slots for in-process worker ---
    def on_in_process_training_finished(self, result):
        logging.info("In-process training worker finished.")
        self.progress_label.setText("Idle")
        self.progress_bar.setValue(0)
        self.integrated_train_button.setEnabled(True)
        self.stop_train_button.setEnabled(False)

        if self.stop_requested:
            StyledMessageBox.warning(self, "Stopped", "Training was stopped by the user.")
        elif result is not False:  # Worker returns False on error
            StyledMessageBox.information(self, "Finished", "Model training completed and saved.")
        else:
            StyledMessageBox.critical(
                self, "Training Failed", "The training process failed. Please check the logs."
            )

        self.training_worker_obj = None
        self.stop_requested = False
        self._reset_training_state()

    def on_worker_error(self, error_str):
        logging.error(f"A worker thread crashed: {error_str}")
        StyledMessageBox.critical(self, "Worker Error", f"A background task failed:\n{error_str}")
        self._reset_training_state()
        # Reset UI state
        self.progress_label.setText("Error")
        self.progress_bar.setValue(0)
        self.integrated_train_button.setEnabled(True)
        self.stop_train_button.setEnabled(False)
        self.training_worker_obj = None
        self.stop_requested = False

    def update_status_label(self, text):
        self.progress_label.setText(text)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def on_chart_update_live(self, update_data):
        # Support both legacy full-payload updates and new incremental updates.
        episode = update_data.get("current_episode")
        if getattr(self, "_live_episode", None) != episode:
            self._live_episode = episode
            self._live_portfolio_history = []
            self._live_trades_df = pd.DataFrame()
            if hasattr(self, "kline_chart"):
                self.kline_chart.clear_overlays()

        portfolio_changed = False
        if "portfolio_history" in update_data:
            self._live_portfolio_history = list(update_data.get("portfolio_history") or [])
            portfolio_changed = True
        else:
            portfolio_point = update_data.get("portfolio_point")
            portfolio_index = update_data.get("portfolio_index")
            if portfolio_point is not None:
                try:
                    portfolio_point = float(portfolio_point)
                except (TypeError, ValueError):
                    portfolio_point = None
            if portfolio_point is not None:
                if portfolio_index is None:
                    self._live_portfolio_history.append(portfolio_point)
                    portfolio_changed = True
                else:
                    try:
                        portfolio_index = int(portfolio_index)
                    except (TypeError, ValueError):
                        portfolio_index = None
                    if portfolio_index is not None:
                        if portfolio_index < 0:
                            portfolio_index = 0
                        if portfolio_index < len(self._live_portfolio_history):
                            self._live_portfolio_history[portfolio_index] = portfolio_point
                        else:
                            filler = (
                                self._live_portfolio_history[-1]
                                if self._live_portfolio_history
                                else portfolio_point
                            )
                            while len(self._live_portfolio_history) < portfolio_index:
                                self._live_portfolio_history.append(filler)
                            self._live_portfolio_history.append(portfolio_point)
                        portfolio_changed = True

        trades_changed = False
        if "trades" in update_data:
            trades = update_data.get("trades")
            if hasattr(trades, "empty"):
                self._live_trades_df = trades
                trades_changed = True
        else:
            new_trades = update_data.get("new_trades") or []
            if new_trades:
                df_new = pd.DataFrame(new_trades)
                if not df_new.empty and "timestamp" in df_new.columns:
                    df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], errors="coerce")
                    df_new = df_new.dropna(subset=["timestamp"]).set_index("timestamp")
                if not df_new.empty:
                    if getattr(self, "_live_trades_df", None) is None or self._live_trades_df.empty:
                        self._live_trades_df = df_new.sort_index()
                    else:
                        self._live_trades_df = pd.concat(
                            [self._live_trades_df, df_new]
                        ).sort_index()
                    trades_changed = True

        if portfolio_changed:
            self.kline_chart.update_portfolio_curve(self._live_portfolio_history)
        if trades_changed:
            self.kline_chart.update_trades(self._live_trades_df)

    # --- Data Management Handlers ---
    def handle_clear_optimization_history(self):
        # The custom question box returns True on 'Yes', so we check for that directly.
        if StyledMessageBox.question(
            self, "确认操作", "您确定要删除所有优化历史记录吗？\n此操作无法撤销。"
        ):
            files_to_delete = [
                STUDY_DB_PATH,
                PROGRESS_FILE,
                BEST_PARAMS_PATH,
                LEGACY_STUDY_DB_PATH,
                LEGACY_PROGRESS_FILE,
                LEGACY_BEST_PARAMS_PATH,
            ]
            deleted_files = []
            errors = []
            for path in files_to_delete:
                try:
                    if path.exists():
                        path.unlink()
                        try:
                            display_path = str(path.relative_to(PROJECT_ROOT))
                        except ValueError:
                            display_path = str(path)
                        deleted_files.append(display_path)
                except Exception as e:
                    errors.append(f"无法删除 {path}: {e}")

            if not errors:
                StyledMessageBox.information(self, "成功", "成功删除:\n" + "\n".join(deleted_files))
            else:
                StyledMessageBox.critical(self, "错误", "\n".join(errors))

    def handle_delete_all_models(self):
        # The custom question box returns True on 'Yes', so we check for that directly.
        if StyledMessageBox.question(
            self, "确认操作", "您确定要删除所有已保存的模型吗？\n此操作无法撤销。"
        ):
            model_dir = MODELS_DIR
            if not model_dir.is_dir():
                StyledMessageBox.information(self, "提示", "模型目录未找到。")
                return

            deleted_count = 0
            errors = []
            for file_path in model_dir.glob("*.zip"):
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"无法删除 {file_path.name}: {e}")

            if not errors:
                StyledMessageBox.information(self, "成功", f"成功删除 {deleted_count} 个模型。")
            else:
                StyledMessageBox.critical(
                    self,
                    "错误",
                    f"已删除 {deleted_count} 个模型，但遇到以下错误：\n" + "\n".join(errors),
                )

    def _reset_training_state(self):
        """重置训练状态变量"""
        self.integrated_training_active = False
        self.training_stage = None

    def apply_stylesheet(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stylesheet_path = os.path.join(current_dir, "styles", "dark_theme.qss")
        try:
            with open(stylesheet_path, "r", encoding="utf-8") as f:
                stylesheet = f.read()
            icon_dir = os.path.join(current_dir, "styles", "icons").replace("\\", "/")
            stylesheet = stylesheet.replace('url("icons/', f'url("{icon_dir}/')
            self.setStyleSheet(stylesheet)
        except FileNotFoundError:
            logger.warning("Stylesheet not found: %s", stylesheet_path)

        for page in [self.settings_page, self.models_page]:
            for label in page.findChildren(QLabel):
                if "<h3>" in label.text():
                    label.setText(label.text().replace("<h3>", "").replace("</h3>", ""))
                    label.setProperty("cssClass", "title")

    def _setup_animations(self):
        self.animations = {}
        for i in range(self.stacked_widget.count()):
            widget = self.stacked_widget.widget(i)
            anim = QPropertyAnimation(widget, b"geometry")
            anim.setDuration(300)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            self.animations[widget] = anim

    def switch_page(self, new_page):
        current_widget = self.stacked_widget.currentWidget()
        if new_page == current_widget:
            return
        current_index = self.stacked_widget.currentIndex()
        new_index = self.stacked_widget.indexOf(new_page)
        if new_index > current_index:
            new_page.setGeometry(self.width(), 0, new_page.width(), new_page.height())
        else:
            new_page.setGeometry(-self.width(), 0, new_page.width(), new_page.height())
        self.stacked_widget.setCurrentWidget(new_page)
        anim_current = self.animations[current_widget]
        if new_index > current_index:
            anim_current.setEndValue(
                QRect(-self.width(), 0, current_widget.width(), current_widget.height())
            )
        else:
            anim_current.setEndValue(
                QRect(self.width(), 0, current_widget.width(), current_widget.height())
            )
        anim_new = self.animations[new_page]
        anim_new.setEndValue(QRect(0, 0, new_page.width(), new_page.height()))
        anim_current.start()
        anim_new.start()

    def _set_cursor_style(self, pos):
        rect = self.rect()
        on_top = pos.y() < self.border_width
        on_bottom = pos.y() > rect.height() - self.border_width
        on_left = pos.x() < self.border_width
        on_right = pos.x() > rect.width() - self.border_width

        if on_top and on_left:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif on_top and on_right:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif on_bottom and on_left:
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif on_bottom and on_right:
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif on_top:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif on_bottom:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif on_left:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif on_right:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            rect = self.rect()

            on_top = pos.y() < self.border_width
            on_bottom = pos.y() > rect.height() - self.border_width
            on_left = pos.x() < self.border_width
            on_right = pos.x() > rect.width() - self.border_width

            if on_top or on_bottom or on_left or on_right:
                self.resizing = True
                self.resize_edge = (on_top, on_bottom, on_left, on_right)
                self.old_pos = event.globalPosition().toPoint()
            else:
                self.moving = True
                self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self.resizing:
            global_pos = event.globalPosition().toPoint()
            delta = global_pos - self.old_pos
            self.old_pos = global_pos

            rect = self.geometry()
            top, bottom, left, right = self.resize_edge

            if top:
                rect.setTop(rect.top() + delta.y())
            if bottom:
                rect.setBottom(rect.bottom() + delta.y())
            if left:
                rect.setLeft(rect.left() + delta.x())
            if right:
                rect.setRight(rect.right() + delta.x())

            if rect.width() < self.minimumWidth():
                rect.setWidth(self.minimumWidth())
            if rect.height() < self.minimumHeight():
                rect.setHeight(self.minimumHeight())

            self.setGeometry(rect)
        elif self.moving:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPosition().toPoint()
        else:
            self._set_cursor_style(pos)

    def mouseReleaseEvent(self, event):
        self.moving = False
        self.resizing = False
        self.old_pos = None
        self.resize_edge = None
        self.unsetCursor()

    def create_chat_page(self):
        """创建聊天页面，集成现有的ChatWidget"""
        self.chat_page = QWidget()
        layout = QVBoxLayout(self.chat_page)
        layout.setContentsMargins(0, 0, 0, 0)

        # 使用原有的ChatWidget
        self.chat_widget = ChatWidget()
        layout.addWidget(self.chat_widget)

    def closeEvent(self, event):
        """Ensure background processes are terminated when the window is closed."""
        logging.info("Close event triggered. Stopping any running worker.")
        self.stop_process()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


RealTimeDashboard = ModernDashboard

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dashboard = ModernDashboard()
    dashboard.show()
    sys.exit(app.exec())
