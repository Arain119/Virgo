"""Chat panel widget for the Qt dashboard.

Combines message rendering, conversation history, LLM streaming, and optional
instruction execution used by the in-app assistant.
"""

import asyncio
import logging
import os

import markdown
from pymdownx.emoji import to_svg, twemoji
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...utils.chat_history_manager import ChatHistoryManager
from ...utils.deepseek_client import DeepSeekR1Client, DeepSeekWorker
from ...utils.instruction_executor import InstructionController
from ...utils.paths import PROJECT_ROOT
from .adaptive_input import AdaptiveInputWidget
from .history_sidebar import HistorySidebar
from .message_bubble import MessageBubble

logger = logging.getLogger(__name__)


class TitleGenerationWorker(QThread):
    """专门用于生成对话标题的工作线程"""

    title_generated = pyqtSignal(str, str)  # conversation_id, title
    title_failed = pyqtSignal(str, str)  # conversation_id, user_message

    def __init__(
        self,
        client: DeepSeekR1Client,
        messages: list,
        conversation_id: str,
        fallback_text: str = "",
    ):
        super().__init__()
        self.client = client
        self.messages = messages
        self.conversation_id = conversation_id
        self.generated_title = ""
        self.user_message = (fallback_text or "")[:50]

        # 如果没有提供备用文本，则尝试从消息中提取
        if not self.user_message:
            for msg in messages:
                if msg.get("role") == "user":
                    self.user_message = msg.get("content", "")[:50]
                    break

    def run(self):
        """在线程中运行异步标题生成"""
        try:
            # 设置事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 连接信号来捕获生成的内容
            self.client.answer_chunk.connect(self.on_title_chunk)
            self.client.response_complete.connect(self.on_title_complete)
            self.client.error_occurred.connect(self.on_title_error)

            # 运行异步请求
            loop.run_until_complete(
                self.client.chat_stream(
                    self.messages,
                    max_tokens=100,
                    temperature=0.3,  # 降低温度以获得更稳定的标题
                    enable_thinking=False,
                )
            )

        except Exception as e:
            logger.exception("标题生成 worker 线程执行失败: %s", e)
            self.title_failed.emit(self.conversation_id, self.user_message)
        finally:
            # 断开信号连接
            try:
                self.client.answer_chunk.disconnect(self.on_title_chunk)
                self.client.response_complete.disconnect(self.on_title_complete)
                self.client.error_occurred.disconnect(self.on_title_error)
            except (TypeError, RuntimeError):
                pass

    def on_title_chunk(self, chunk):
        """接收标题内容块"""
        self.generated_title += chunk

    def on_title_complete(self):
        """标题生成完成"""
        if self.generated_title.strip():
            self.title_generated.emit(self.conversation_id, self.generated_title.strip())
        else:
            self.title_failed.emit(self.conversation_id, self.user_message)

    def on_title_error(self, error):
        """标题生成错误"""
        logger.warning("标题生成 API 错误: %s", error)
        self.title_failed.emit(self.conversation_id, self.user_message)


class ChatWidget(QWidget):
    """主聊天界面"""

    # 定义线程安全的信号 - 用于工作线程向主线程传递数据
    iteration_started = pyqtSignal(int)
    thinking_chunk_received = pyqtSignal(str, int)
    answer_chunk_received = pyqtSignal(str, int)
    thinking_completed = pyqtSignal(int)
    instruction_completed = pyqtSignal(str)
    instruction_failed = pyqtSignal(str)
    iteration_complete_signal = pyqtSignal(int, str, str)  # iteration, thinking, answer

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_key = os.environ.get("SILICONFLOW_API_KEY") or ""
        self.api_url = "https://api.siliconflow.cn/v1"

        # 创建一个共享的Markdown实例
        self.md = markdown.Markdown(
            extensions=[
                "extra",
                "pymdownx.superfences",
                "pymdownx.emoji",
                "pymdownx.highlight",
            ],
            extension_configs={
                "pymdownx.emoji": {"emoji_index": twemoji, "emoji_generator": to_svg},
                "pymdownx.highlight": {
                    "css_class": "highlight",
                    "use_pygments": True,
                    "noclasses": False,
                },
            },
        )

        # 初始化管理器和客户端
        self.history_manager = ChatHistoryManager()
        self.deepseek_client = DeepSeekR1Client(self.api_key, self.api_url)
        self.current_conversation_id = None

        # 当前消息状态
        self.current_thinking = ""
        self.current_answer = ""
        self.current_message_bubble = None
        self.deepseek_worker = None

        # 使用流式InstructionController处理所有指令逻辑
        self.instruction_controller = InstructionController(self.deepseek_client)
        # 设置线程安全的流式回调 - 通过信号发射到主线程
        self.instruction_controller.set_streaming_callbacks(
            self.emit_thinking_chunk,
            self.emit_answer_chunk,
            self.emit_thinking_complete,
            self.emit_iteration_start,
            self.emit_iteration_complete,
        )

        # 多轮交互状态管理
        self.current_iteration = 0
        self.iteration_bubbles = {}  # 存储每轮交互的消息气泡
        self.final_response_parts = []  # 收集所有轮次的回复

        # 侧栏显示状态
        self.sidebar_visible = False

        # 文档相关
        # The project root is used so the assistant can read README.md and any user-provided docs.
        self.docs_path = str(PROJECT_ROOT)
        self.supported_formats = [".txt", ".md", ".docx"]
        self.document_list = []

        self.setup_ui()
        self.connect_signals()
        self.load_documents()

        # 不创建默认对话，等用户发送消息时再创建
        # self.start_new_conversation()

    def setup_ui(self):
        """设置用户界面"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 左侧：历史记录侧栏
        self.history_sidebar = HistorySidebar(self.history_manager)
        self.history_sidebar.setFixedWidth(200)

        # 右侧：聊天区域
        chat_area = QWidget()
        chat_area.setObjectName("chatMainArea")
        chat_layout = QVBoxLayout(chat_area)
        chat_layout.setContentsMargins(20, 10, 20, 20)
        chat_layout.setSpacing(8)

        # 添加聊天顶部工具栏
        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("chatToolbar")
        toolbar_layout = QHBoxLayout(toolbar_frame)
        toolbar_layout.setContentsMargins(5, 5, 5, 5)

        # 汉堡菜单按钮
        self.hamburger_button = QToolButton()
        self.hamburger_button.setObjectName("hamburgerButton")
        self.hamburger_button.setText("☰")
        self.hamburger_button.setFixedSize(32, 32)
        self.hamburger_button.setCheckable(True)
        self.hamburger_button.clicked.connect(self.toggle_sidebar)

        # 标题标签
        chat_title = QLabel("Virgo Trader 智能助手")
        chat_title.setObjectName("chatTitle")
        chat_title.setStyleSheet("color: #bd93f9; font-size: 16px; font-weight: bold;")

        toolbar_layout.addWidget(self.hamburger_button)
        toolbar_layout.addWidget(chat_title)
        toolbar_layout.addStretch()

        chat_layout.addWidget(toolbar_frame)

        # 聊天消息滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setObjectName("chatScrollArea")

        # 消息容器
        self.messages_container = QWidget()
        self.messages_container.setObjectName("messagesContainer")
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(15)
        self.messages_layout.addStretch()  # 将消息推到底部

        self.scroll_area.setWidget(self.messages_container)

        # 自适应输入区域
        self.adaptive_input = AdaptiveInputWidget()

        # 智能指令状态显示
        self.instruction_status_label = QLabel()
        self.instruction_status_label.setObjectName("instructionStatus")
        self.instruction_status_label.hide()  # 默认隐藏
        self.instruction_status_label.setStyleSheet("""
            QLabel#instructionStatus {
                color: #bd93f9;
                background-color: rgba(189, 147, 249, 0.1);
                border: 1px solid rgba(189, 147, 249, 0.3);
                border-radius: 5px;
                padding: 8px 12px;
                font-size: 12px;
                margin: 5px 0px;
            }
        """)

        chat_layout.addWidget(self.scroll_area, 1)
        chat_layout.addWidget(self.instruction_status_label)
        chat_layout.addWidget(self.adaptive_input)

        # 直接添加到布局（移除分割器）
        layout.addWidget(self.history_sidebar)
        layout.addWidget(chat_area)

        # 初始状态隐藏侧栏
        self.history_sidebar.hide()

    def connect_signals(self):
        """连接信号和槽"""
        # 自适应输入框信号
        self.adaptive_input.send_requested.connect(self.on_send_requested)
        self.adaptive_input.stop_generation.connect(self.on_stop_generation)

        # 移除直接的DeepSeek客户端信号连接，统一通过InstructionController处理
        # 这样避免多重信号连接导致的字符重复显示问题

        # 历史记录侧栏信号
        self.history_sidebar.conversation_selected.connect(self.load_conversation)
        self.history_sidebar.new_conversation_requested.connect(self.start_new_conversation)
        self.history_sidebar.conversation_deleted.connect(self.on_conversation_deleted)

        # 连接线程安全的信号到主线程槽函数
        self.iteration_started.connect(self.on_iteration_start_safe)
        self.thinking_chunk_received.connect(self.on_thinking_chunk_safe)
        self.answer_chunk_received.connect(self.on_answer_chunk_safe)
        self.thinking_completed.connect(self.on_thinking_complete_safe)
        self.instruction_completed.connect(self._on_instruction_completed)
        self.instruction_failed.connect(self._on_instruction_failed)
        self.iteration_complete_signal.connect(self.on_iteration_complete_safe)

        # 线程管理
        self.title_worker = None

    def load_documents(self):
        """加载文档列表"""
        self.document_list = []

        if not os.path.exists(self.docs_path):
            return

        # 扫描docs文件夹
        for filename in sorted(os.listdir(self.docs_path)):
            file_path = os.path.join(self.docs_path, filename)

            # 只处理文件，跳过目录
            if not os.path.isfile(file_path):
                continue

            # 检查文件扩展名
            _, ext = os.path.splitext(filename)
            if ext.lower() in self.supported_formats:
                # 添加图标
                if ext.lower() == ".md":
                    icon = "📝"
                elif ext.lower() == ".txt":
                    icon = "📄"
                elif ext.lower() == ".docx":
                    icon = "📋"
                else:
                    icon = "📄"

                self.document_list.append(
                    {"filename": filename, "filepath": file_path, "display": f"{icon} {filename}"}
                )

    def extract_document_content(self, file_path):
        """提取文档内容"""
        try:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

            elif ext == ".md":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

            elif ext == ".docx":
                try:
                    from docx import Document

                    doc = Document(file_path)
                    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                    return "\n".join(paragraphs)
                except ImportError:
                    return "需要安装python-docx库来读取Word文档"

        except Exception as e:
            return f"读取文件时出错: {str(e)}"

        return ""

    def on_input_changed(self):
        """输入框内容变化时的处理"""
        # 检测#号触发文档选择
        text = self.input_field.toPlainText()
        cursor = self.input_field.textCursor()
        current_pos = cursor.position()

        # 查找最近的#号位置
        hash_pos = text.rfind("#", 0, current_pos)
        if hash_pos != -1:
            # 检查#号后面是否只有字母数字或为空
            after_hash = text[hash_pos + 1 : current_pos]
            if after_hash == "" or after_hash.replace(" ", "").isalnum():
                self.show_document_selector(hash_pos, after_hash)
            else:
                self.hide_document_selector()
        else:
            self.hide_document_selector()

        # 支持Ctrl+Enter发送
        if QApplication.keyboardModifiers() == Qt.KeyboardModifier.ControlModifier:
            cursor = self.input_field.textCursor()
            if cursor.hasSelection() or cursor.atEnd():
                self.send_message()

    def show_document_selector(self, hash_pos, search_term):
        """显示文档选择器"""
        if not hasattr(self, "document_selector"):
            self.create_document_selector()

        # 过滤文档列表
        filtered_docs = []
        search_lower = search_term.lower()
        for doc in self.document_list:
            if search_lower in doc["filename"].lower():
                filtered_docs.append(doc)

        # 更新选择器内容
        self.document_selector.clear()
        for doc in filtered_docs[:10]:  # 最多显示10个
            item = QListWidgetItem(doc["display"])
            item.setData(Qt.ItemDataRole.UserRole, doc)
            self.document_selector.addItem(item)

        if filtered_docs:
            # 计算位置并显示
            cursor = self.input_field.textCursor()
            cursor.setPosition(hash_pos)
            rect = self.input_field.cursorRect(cursor)
            global_pos = self.input_field.mapToGlobal(rect.bottomLeft())

            self.document_selector.move(global_pos)
            self.document_selector.resize(300, min(200, len(filtered_docs) * 25 + 10))
            self.document_selector.show()
            self.document_selector.raise_()
        else:
            self.hide_document_selector()

    def hide_document_selector(self):
        """隐藏文档选择器"""
        if hasattr(self, "document_selector"):
            self.document_selector.hide()

    def create_document_selector(self):
        """创建文档选择器"""
        self.document_selector = QListWidget(self)
        self.document_selector.setObjectName("documentSelector")
        self.document_selector.setWindowFlags(Qt.WindowType.Popup)
        self.document_selector.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.document_selector.itemClicked.connect(self.on_document_selector_clicked)

        # 设置样式
        self.document_selector.setStyleSheet("""
            QListWidget#documentSelector {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
                color: white;
                font-size: 12px;
            }
            QListWidget#documentSelector::item {
                padding: 5px;
                border-bottom: 1px solid #444;
            }
            QListWidget#documentSelector::item:hover {
                background-color: #404040;
            }
            QListWidget#documentSelector::item:selected {
                background-color: #bd93f9;
            }
        """)

    def on_document_selector_clicked(self, item):
        """处理文档选择器点击"""
        doc_data = item.data(Qt.ItemDataRole.UserRole)
        if not doc_data:
            return

        # 读取文档内容
        content = self.extract_document_content(doc_data["filepath"])
        if not content:
            return

        # 替换输入框中的内容
        text = self.input_field.toPlainText()
        cursor = self.input_field.textCursor()
        current_pos = cursor.position()

        # 找到#号位置
        hash_pos = text.rfind("#", 0, current_pos)
        if hash_pos != -1:
            # 格式化文档内容
            formatted_prompt = f"""请基于以下文档内容回答问题：

=== 文档：{doc_data["filename"]} ===
{content}
=== 文档内容结束 ===

现在请回答我的问题："""

            # 替换从#号到当前位置的内容
            new_text = text[:hash_pos] + formatted_prompt + text[current_pos:]
            self.input_field.setPlainText(new_text)

            # 设置光标到末尾
            cursor = self.input_field.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.input_field.setTextCursor(cursor)

        # 隐藏选择器并聚焦输入框
        self.hide_document_selector()
        self.input_field.setFocus()

    def on_send_requested(self, message_text: str):
        """处理自适应输入框的发送请求 - 使用InstructionController"""
        if not message_text.strip() or self.deepseek_worker:
            return

        # 如果没有当前对话，则创建新对话
        if not self.current_conversation_id:
            self.current_conversation_id = self.history_manager.create_conversation()

        # 添加用户消息
        user_bubble = MessageBubble("user", message_text, md_instance=self.md)
        self.add_message_bubble(user_bubble)

        # 保存用户消息到历史
        self.history_manager.add_message(self.current_conversation_id, "user", message_text)

        # 创建AI回复消息泡
        self.current_message_bubble = MessageBubble("assistant", "", md_instance=self.md)
        self.add_message_bubble(self.current_message_bubble)

        # 重置当前状态
        self.current_thinking = ""
        self.current_answer = ""

        # 保存用户消息文本供后续AI命名使用
        self.first_user_message = message_text

        # 获取完整的对话历史
        messages = self.history_manager.get_conversation_messages(self.current_conversation_id)

        # 创建并启动异步任务处理指令
        self._start_instruction_processing_task(messages)

        # 更新历史侧栏
        self.history_sidebar.refresh_conversations()

    def _start_instruction_processing_task(self, messages: list):
        """启动主线程异步指令处理任务"""

        # 保存消息
        self.pending_messages = messages

        # 使用QTimer在主线程中异步执行
        self.instruction_timer = QTimer()
        self.instruction_timer.setSingleShot(True)
        self.instruction_timer.timeout.connect(self._process_instruction_in_main_thread)
        self.instruction_timer.start(50)  # 50ms后执行，给UI时间更新

    def _process_instruction_in_main_thread(self):
        """在主线程中处理指令"""
        try:
            import threading

            # 创建并启动专门的异步处理线程
            self.async_thread = threading.Thread(target=self._run_async_instruction_processing)
            self.async_thread.daemon = True
            self.async_thread.start()

            # 使用QTimer定期检查结果
            self.result_check_timer = QTimer()
            self.result_check_timer.timeout.connect(self._check_instruction_result)
            self.result_check_timer.start(100)  # 每100ms检查一次

            # 初始化结果变量
            self.instruction_result = None
            self.instruction_error = None

        except Exception as e:
            logger.error(f"启动指令处理失败: {e}")
            self._on_instruction_failed(str(e))

    def _run_async_instruction_processing(self):
        """在独立线程中运行异步指令处理"""
        try:
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 运行异步处理
            result = loop.run_until_complete(self._async_process_instruction())
            self.instruction_result = result

        except Exception as e:
            logger.error(f"异步指令处理失败: {e}")
            self.instruction_error = str(e)
        finally:
            # 清理事件循环
            if loop and not loop.is_closed():
                loop.close()

    def _check_instruction_result(self):
        """检查指令处理结果"""
        if self.instruction_result is not None:
            # 停止检查计时器
            if hasattr(self, "result_check_timer"):
                self.result_check_timer.stop()
                self.result_check_timer = None

            # 处理成功结果
            result = self.instruction_result
            self.instruction_result = None
            self._on_instruction_completed(result)

        elif self.instruction_error is not None:
            # 停止检查计时器
            if hasattr(self, "result_check_timer"):
                self.result_check_timer.stop()
                self.result_check_timer = None

            # 处理错误
            error = self.instruction_error
            self.instruction_error = None
            self._on_instruction_failed(error)

    async def _async_process_instruction(self):
        """异步处理指令"""
        try:
            # 格式化消息以匹配API
            api_messages = []
            for msg in self.pending_messages:
                api_message = {"role": msg["role"], "content": msg["content"]}
                api_messages.append(api_message)

            # 执行指令处理（使用流式方法）
            final_response = await self.instruction_controller.process_message_with_streaming(
                api_messages
            )

            return final_response

        except Exception as e:
            logger.error(f"异步指令处理失败: {e}")
            raise e

    def _check_async_task(self, task):
        """检查异步任务状态"""
        if task.done():
            # 停止检查计时器
            if hasattr(self, "check_timer"):
                self.check_timer.stop()
                self.check_timer = None

            try:
                # 获取结果
                final_response = task.result()
                self._on_instruction_completed(final_response)

            except Exception as e:
                self._on_instruction_failed(str(e))

    def _on_instruction_completed(self, final_response: str):
        """指令处理完成"""
        try:
            # 清理最终回复中的指令标记
            clean_response = self.clean_instruction_markers(final_response)

            # 更新消息气泡
            if self.current_message_bubble:
                self.current_message_bubble.update_answer(clean_response)
                self.current_message_bubble.set_streaming_complete()

            # 保存最终的AI回复到历史 - 这部分逻辑已移至 on_iteration_complete_safe
            # 但我们仍然需要保存最后 analyze 阶段的回复
            if self.current_conversation_id:
                self.history_manager.add_message(
                    self.current_conversation_id,
                    "assistant",
                    clean_response,
                    thinking=self.current_thinking,
                )

            # 检查是否需要生成对话标题
            if hasattr(self, "first_user_message") and self.first_user_message:
                self.generate_conversation_title(self.first_user_message, clean_response)
                self.first_user_message = None

        except Exception as e:
            logger.error(f"处理完成回调失败: {e}")
        finally:
            self._cleanup_instruction_processing()

    def _on_instruction_failed(self, error_message: str):
        """指令处理失败"""
        logger.error(f"指令处理失败: {error_message}")
        if self.current_message_bubble:
            self.current_message_bubble.show_error(f"处理失败: {error_message}")
        self._cleanup_instruction_processing()

    def _cleanup_instruction_processing(self):
        """清理指令处理状态"""
        # 清理计时器
        if hasattr(self, "instruction_timer"):
            self.instruction_timer.stop()
            self.instruction_timer = None
        if hasattr(self, "check_timer"):
            self.check_timer.stop()
            self.check_timer = None

        # 隐藏状态标签
        self.instruction_status_label.hide()
        # 重置状态
        self.current_message_bubble = None
        # 通知输入框发送完成
        self.adaptive_input.set_sending_complete()

    def send_message(self):
        """发送消息 - 兼容性方法"""
        # 从自适应输入框获取文本并发送
        message_text = self.adaptive_input.get_text().strip()
        if message_text:
            self.on_send_requested(message_text)

    def on_thinking_chunk(self, chunk):
        """接收思考过程块"""
        self.current_thinking += chunk
        if self.current_message_bubble:
            self.current_message_bubble.update_thinking(self.current_thinking)

    def on_answer_chunk(self, chunk):
        """接收答案块"""
        self.current_answer += chunk
        if self.current_message_bubble:
            # 清理指令标记后再显示给用户
            clean_answer = self.clean_instruction_markers(self.current_answer)
            self.current_message_bubble.update_answer(clean_answer)

        # 检测智能指令并显示状态
        self.detect_and_show_instruction_status()

        # 自动滚动到底部
        QTimer.singleShot(50, self.scroll_to_bottom)

    def on_thinking_complete(self):
        """思考完成"""
        if self.current_message_bubble:
            # 立即收起思考框并改名为"Thought"
            self.current_message_bubble.on_thinking_complete()

    def on_response_complete(self):
        """响应完成 - 简化版本，移除复杂的指令处理逻辑"""
        # 清理最终答案中的指令标记
        final_answer = self.clean_instruction_markers(self.current_answer)

        # 保存清理后的AI回复到历史
        if self.current_conversation_id:
            self.history_manager.add_message(
                self.current_conversation_id, "assistant", final_answer, self.current_thinking
            )

        # 更新消息气泡显示最终清理后的内容
        if self.current_message_bubble:
            self.current_message_bubble.update_answer(final_answer)
            self.current_message_bubble.set_streaming_complete()

        # 隐藏指令状态
        self.instruction_status_label.hide()

        # 检查是否需要生成对话标题（仅对新对话的第一次回复）
        if hasattr(self, "first_user_message") and self.first_user_message:
            self.generate_conversation_title(self.first_user_message, final_answer)
            self.first_user_message = None  # 重置，避免重复生成

        # 重置状态
        self.current_message_bubble = None

    def on_error_occurred(self, error_message):
        """处理错误"""
        if self.current_message_bubble:
            self.current_message_bubble.show_error(f"发生错误: {error_message}")

    def on_stop_generation(self):
        """处理停止生成请求"""
        if self.deepseek_worker:
            # 终止工作线程
            self.deepseek_worker.terminate()
            self.deepseek_worker.wait()  # 等待线程完全停止
            self.deepseek_worker = None

            # 更新当前消息气泡状态
            if self.current_message_bubble:
                current_content = self.current_answer if self.current_answer else "生成已停止"
                self.current_message_bubble.update_answer(current_content)
                self.current_message_bubble.set_streaming_complete()

                # 保存当前内容到历史（如果有内容的话）
                if self.current_conversation_id and self.current_answer:
                    self.history_manager.add_message(
                        self.current_conversation_id,
                        "assistant",
                        self.current_answer,
                        self.current_thinking,
                    )

            # 重置状态
            self.current_message_bubble = None
            self.current_thinking = ""
            self.current_answer = ""

            # 通知输入框发送完成
            self.adaptive_input.set_sending_complete()

    def on_worker_finished(self):
        """工作线程完成"""
        self.deepseek_worker = None
        # 通知自适应输入框发送完成
        self.adaptive_input.set_sending_complete()

    def toggle_sidebar(self):
        """切换侧栏显示/隐藏"""
        self.sidebar_visible = not self.sidebar_visible

        if self.sidebar_visible:
            # 显示侧栏
            self.history_sidebar.show()
            self.hamburger_button.setChecked(True)
        else:
            # 隐藏侧栏
            self.history_sidebar.hide()
            self.hamburger_button.setChecked(False)

    def add_message_bubble(self, bubble):
        """添加消息泡到界面"""
        # 在stretch之前插入
        count = self.messages_layout.count()
        self.messages_layout.insertWidget(count - 1, bubble)

        # 连接重新生成信号（仅对AI消息）
        if bubble.role == "assistant":
            bubble.regenerate_requested.connect(self.on_regenerate_requested)

        # 滚动到底部
        QTimer.singleShot(100, self.scroll_to_bottom)

    def scroll_to_bottom(self):
        """滚动到底部"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_new_conversation(self):
        """开始新对话"""
        self.current_conversation_id = self.history_manager.create_conversation()
        self.clear_messages()
        self.history_sidebar.refresh_conversations()
        self.history_sidebar.select_conversation(self.current_conversation_id)

    def load_conversation(self, conversation_id):
        """加载对话"""
        logger.debug("ChatWidget.load_conversation: %s", conversation_id)
        self.current_conversation_id = conversation_id
        self.clear_messages()

        messages = self.history_manager.get_conversation_messages(conversation_id)
        logger.debug("Loading %d messages", len(messages))
        for msg in messages:
            if msg["role"] == "user":
                bubble = MessageBubble("user", msg["content"], md_instance=self.md)
            else:
                thinking = msg.get("thinking", "")
                bubble = MessageBubble("assistant", msg["content"], thinking, md_instance=self.md)

            self.add_message_bubble(bubble)

    def clear_messages(self):
        """清空消息"""
        # 移除除了stretch之外的所有widget
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def on_conversation_deleted(self, conversation_id):
        """对话被删除"""
        if conversation_id == self.current_conversation_id:
            # 清空当前对话ID和消息，但不创建新对话
            self.current_conversation_id = None
            self.clear_messages()
        # 可选择切换到最近的其他对话
        grouped_conversations = self.history_manager.get_conversations_by_date()
        # 找到最近的对话
        latest_conv = None
        for group_conversations in grouped_conversations.values():
            if group_conversations:
                latest_conv = group_conversations[0]
                break

        if latest_conv:
            # 切换到最近的对话
            self.load_conversation(latest_conv["id"])
            self.history_sidebar.select_conversation(latest_conv["id"])

    def on_regenerate_requested(self):
        """处理重新生成请求"""
        if self.deepseek_worker:  # 如果正在生成，先停止
            return

        # 找到发起请求的消息气泡
        sender = self.sender()
        if not sender:
            return

        # 获取对话历史，排除最后一个AI回复
        messages = self.history_manager.get_conversation_messages(self.current_conversation_id)
        api_messages = []
        for msg in messages[:-1]:  # 排除最后一个AI消息
            if msg["role"] != "assistant" or msg != messages[-1]:  # 确保不包括要重新生成的消息
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # 如果没有消息历史，无法重新生成
        if not api_messages:
            return

        # 重置消息泡状态
        self.current_message_bubble = sender
        self.current_thinking = ""
        self.current_answer = ""

        # 重置显示
        sender.thinking = ""
        sender.content = ""
        sender.thinking_content.clear()
        sender.answer_content.setText("正在重新生成...")
        sender.loading_label.show()

        # 启动AI请求
        self.deepseek_worker = DeepSeekWorker(self.deepseek_client, api_messages)
        self.deepseek_worker.finished.connect(self.on_worker_finished)
        self.deepseek_worker.start()

        # 设置输入框为发送状态（禁用输入）
        self.adaptive_input.set_enabled(False)

    def generate_conversation_title(self, user_message, ai_response):
        """使用AI生成对话标题 - 使用和主聊天相同的DeepSeek客户端"""
        try:
            # 构建用于生成标题的prompt
            title_system_prompt = (
                "你是一名中文标题助手，请用不超过20个汉字概括对话内容，"
                "不得输出任何说明或标点装饰，只返回最终标题。"
            )
            title_prompt = f"""根据以下对话内容生成一个简洁的中文标题：

用户：{user_message[:200]}
AI：{ai_response[:300]}

请直接返回标题本身。"""

            # 为标题生成单独提供系统提示，避免触发指令流程
            title_messages = [
                {"role": "system", "content": title_system_prompt},
                {"role": "user", "content": title_prompt},
            ]

            # 创建一个独立的DeepSeek客户端实例用于生成标题
            title_client = DeepSeekR1Client(self.api_key, self.api_url)

            # 创建标题生成的worker，使用相同的逻辑
            self.title_worker = TitleGenerationWorker(
                title_client,
                title_messages,
                self.current_conversation_id,
                fallback_text=user_message,
            )
            self.title_worker.title_generated.connect(self.on_title_generated)
            self.title_worker.title_failed.connect(self.on_title_failed)
            self.title_worker.start()

        except Exception as e:
            logger.exception("生成对话标题失败: %s", e)
            # 如果创建 worker 失败，使用用户消息的前几个字作为标题
            fallback_title = user_message[:15] + "..." if len(user_message) > 15 else user_message
            if self.current_conversation_id:
                self.history_manager.update_conversation_title(
                    self.current_conversation_id, fallback_title
                )
                QTimer.singleShot(500, self.history_sidebar.refresh_conversations)

    def on_title_generated(self, conversation_id, title):
        """标题生成成功"""
        if title and conversation_id:
            # 清理标题，移除可能的引号和多余字符
            title = title.replace('"', "").replace("'", "").replace("标题：", "").strip()

            # 确保标题不超过20个字符
            if len(title) > 20:
                title = title[:17] + "..."

            # 更新对话标题
            self.history_manager.update_conversation_title(conversation_id, title)
            # 刷新历史侧栏显示
            QTimer.singleShot(500, self.history_sidebar.refresh_conversations)

    def on_title_failed(self, conversation_id, user_message):
        """标题生成失败，使用备用方案"""
        logger.info("标题生成失败，使用备用方案")
        fallback_title = user_message[:15] + "..." if len(user_message) > 15 else user_message
        if conversation_id:
            self.history_manager.update_conversation_title(conversation_id, fallback_title)
            QTimer.singleShot(500, self.history_sidebar.refresh_conversations)

    def clean_instruction_markers(self, text):
        """清理指令标记，只显示干净的内容给用户"""
        import re

        # 移除指令标记
        clean_text = re.sub(r"\[SCAN_DIR.*?\]", "", text)
        clean_text = re.sub(r"\[READ_FILE.*?\]", "", clean_text)
        clean_text = re.sub(r"\[ANALYZE\]", "", clean_text)
        return clean_text.strip()

    def detect_and_show_instruction_status(self):
        """检测指令并显示状态"""
        if "[SCAN_DIR" in self.current_answer:
            self.instruction_status_label.setText("🔍 正在分析项目...")
            self.instruction_status_label.show()
        elif "[READ_FILE" in self.current_answer:
            self.instruction_status_label.setText("📄 正在读取文件...")
            self.instruction_status_label.show()
        elif "[ANALYZE]" in self.current_answer:
            self.instruction_status_label.hide()

    # 移除execute_instruction_and_continue方法 - 现在使用InstructionController统一处理

    def on_iteration_start(self, iteration_count: int):
        """每轮AI交互开始时调用"""
        logger.info(f"开始第 {iteration_count} 轮AI交互")
        # 如果不是第一轮，为当前轮创建新的消息区域
        if iteration_count > 1:
            # 为新轮次创建分隔符或新的消息泡
            iteration_bubble = MessageBubble(
                "assistant", f"--- 第 {iteration_count} 轮分析 ---", md_instance=self.md
            )
            iteration_bubble.setObjectName("iterationSeparator")
            self.add_message_bubble(iteration_bubble)
            self.current_message_bubble = iteration_bubble

            # 重置当前状态
            self.current_thinking = ""
            self.current_answer = ""

    def on_thinking_chunk_with_iteration(self, chunk: str, iteration: int):
        """带迭代信息的思考块回调"""
        logger.debug(f"第 {iteration} 轮 - 思考块: {chunk[:50]}...")
        self.current_thinking += chunk
        if self.current_message_bubble:
            self.current_message_bubble.update_thinking(self.current_thinking)

    def on_answer_chunk_with_iteration(self, chunk: str, iteration: int):
        """带迭代信息的答案块回调"""
        logger.debug(f"第 {iteration} 轮 - 答案块: {chunk[:50]}...")
        self.current_answer += chunk
        if self.current_message_bubble:
            # 清理指令标记后再显示给用户
            clean_answer = self.clean_instruction_markers(self.current_answer)
            self.current_message_bubble.update_answer(clean_answer)

        # 检测智能指令并显示状态
        self.detect_and_show_instruction_status_with_iteration(iteration)

        # 自动滚动到底部
        QTimer.singleShot(50, self.scroll_to_bottom)

    def on_thinking_complete_with_iteration(self, iteration: int):
        """带迭代信息的思考完成回调"""
        logger.info(f"第 {iteration} 轮思考完成")
        if self.current_message_bubble:
            self.current_message_bubble.on_thinking_complete()

    def detect_and_show_instruction_status_with_iteration(self, iteration: int):
        """检测指令并显示状态（带迭代信息）"""
        status_text = ""
        if "[SCAN_DIR" in self.current_answer:
            status_text = f"第{iteration}轮: 正在分析项目结构..."
        elif "[READ_FILE" in self.current_answer:
            status_text = f"第{iteration}轮: 正在读取文件..."
        elif "[ANALYZE]" in self.current_answer:
            status_text = f"第{iteration}轮: 分析完成，正在整理回复..."
            # 最终轮，短暂显示后隐藏
            QTimer.singleShot(2000, self.instruction_status_label.hide)

        if status_text:
            self.instruction_status_label.setText(status_text)
            self.instruction_status_label.show()

    # ============= 线程安全的回调方法 =============
    def emit_iteration_start(self, iteration_count: int):
        """线程安全的迭代开始回调 - 发射信号到主线程"""
        self.iteration_started.emit(iteration_count)

    def emit_thinking_chunk(self, chunk: str, iteration: int):
        """线程安全的思考块回调 - 发射信号到主线程"""
        self.thinking_chunk_received.emit(chunk, iteration)

    def emit_answer_chunk(self, chunk: str, iteration: int):
        """线程安全的答案块回调 - 发射信号到主线程"""
        self.answer_chunk_received.emit(chunk, iteration)

    def emit_thinking_complete(self, iteration: int):
        """线程安全的思考完成回调 - 发射信号到主线程"""
        self.thinking_completed.emit(iteration)

    def emit_iteration_complete(self, iteration: int, thinking: str, answer: str):
        """线程安全的迭代完成回调 - 发射信号到主线程"""
        self.iteration_complete_signal.emit(iteration, thinking, answer)

    # ============= 主线程中的安全槽函数 =============
    def on_iteration_start_safe(self, iteration_count: int):
        """主线程中的安全迭代开始处理"""
        logger.info(f"开始第 {iteration_count} 轮AI交互")
        # 如果不是第一轮，为当前轮创建新的消息区域
        if iteration_count > 1:
            # 创建支持完整thinking交互的消息泡，而不是简单的分隔符
            iteration_bubble = MessageBubble(
                "assistant", "", md_instance=self.md
            )  # 空内容，等待thinking和answer
            self.add_message_bubble(iteration_bubble)
            self.current_message_bubble = iteration_bubble

            # 重置当前状态
            self.current_thinking = ""
            self.current_answer = ""

            # 新轮次开始时立即滚动到底部，确保用户看到新的交互
            QTimer.singleShot(100, self.scroll_to_bottom)

    def on_thinking_chunk_safe(self, chunk: str, iteration: int):
        """主线程中的安全思考块处理"""
        # 移除详细DEBUG日志，只保留必要的处理逻辑
        self.current_thinking += chunk
        if self.current_message_bubble:
            self.current_message_bubble.update_thinking(self.current_thinking)

        # 在多轮交互中，思考内容更新时也要滚动，确保用户能看到最新的思考过程
        if iteration > 1:
            QTimer.singleShot(30, self.scroll_to_bottom)

    def on_answer_chunk_safe(self, chunk: str, iteration: int):
        """主线程中的安全答案块处理"""
        # 移除详细DEBUG日志，只保留必要的处理逻辑
        self.current_answer += chunk
        if self.current_message_bubble:
            # 清理指令标记后再显示给用户
            clean_answer = self.clean_instruction_markers(self.current_answer)
            self.current_message_bubble.update_answer(clean_answer)

        # 检测智能指令并显示状态
        self.detect_and_show_instruction_status_with_iteration(iteration)

        # 自动滚动到底部，在多轮交互中使用更短的延迟确保及时滚动
        if iteration > 1:
            QTimer.singleShot(20, self.scroll_to_bottom)
        else:
            QTimer.singleShot(50, self.scroll_to_bottom)

    def on_thinking_complete_safe(self, iteration: int):
        """主线程中的安全思考完成处理"""
        logger.info(f"第 {iteration} 轮思考完成")
        if self.current_message_bubble:
            self.current_message_bubble.on_thinking_complete()

    def on_iteration_complete_safe(self, iteration: int, thinking: str, answer: str):
        """主线程中的安全迭代完成处理"""
        logger.info(f"第 {iteration} 轮交互完成，保存到历史记录")
        if self.current_conversation_id:
            # 清理指令标记
            clean_answer = self.clean_instruction_markers(answer)
            self.history_manager.add_message(
                self.current_conversation_id, "assistant", clean_answer, thinking
            )

    def closeEvent(self, event):
        """窗口关闭事件，确保清理所有线程"""
        # 清理主工作线程
        if self.deepseek_worker:
            self.deepseek_worker.terminate()
            self.deepseek_worker.wait(3000)  # 等待最多3秒
            self.deepseek_worker = None

        # 清理标题生成线程
        if self.title_worker:
            self.title_worker.terminate()
            self.title_worker.wait(3000)  # 等待最多3秒
            self.title_worker = None

        super().closeEvent(event)

    def keyPressEvent(self, event):
        """键盘事件处理"""
        if (
            event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter
        ) and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.send_message()
        else:
            super().keyPressEvent(event)
