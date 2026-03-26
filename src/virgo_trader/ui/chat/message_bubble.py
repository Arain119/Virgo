"""Message bubble widgets used by the chat UI.

Renders user/assistant messages (including code blocks) with a styled layout
suitable for the dashboard.
"""

import re

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .code_block_widget import CodeBlockWidget


class MessageBubble(QWidget):
    """消息气泡组件，支持文本和代码块的动态渲染"""

    regenerate_requested = pyqtSignal()

    def __init__(self, role: str, content: str, thinking: str = "", md_instance=None):
        super().__init__()
        self.role = role
        self.content = content
        self.thinking = thinking
        self.md = md_instance

        self.thinking_visible = False
        self.is_streaming = role == "assistant" and not content

        self.setup_ui()

        if not self.is_streaming:
            self.update_content()

    def get_table_styles(self) -> str:
        """返回表格的CSS样式字符串"""
        return """
        <style>
        table {
            width: 100%; border-collapse: collapse; margin-top: 10px; margin-bottom: 10px;
            background-color: rgba(40, 42, 54, 0.7); border-radius: 8px; border: 1px solid #6272a4;
        }
        thead { background-color: rgba(68, 71, 90, 0.8); font-weight: bold; }
        th, td { border: 1px solid #44475a; padding: 10px 12px; text-align: left; font-size: 13px; }
        th { color: #bd93f9; }
        tr:nth-child(even) { background-color: rgba(40, 42, 54, 0.9); }
        </style>
        """

    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.main_frame = QFrame()
        self.main_frame.setObjectName(f"messageBubble_{self.role}")
        main_layout = QVBoxLayout(self.main_frame)
        main_layout.setContentsMargins(15, 10, 15, 10)
        main_layout.setSpacing(8)

        if self.role == "assistant":
            self.setup_ai_message(main_layout)
        else:
            self.setup_user_message(main_layout)

        layout.addWidget(self.main_frame)

    def setup_user_message(self, layout):
        """设置用户消息界面"""
        self.content_label = QLabel()
        self.content_label.setWordWrap(True)
        self.content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.content_label.setObjectName("userMessageContent")
        self.content_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.content_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.content_label.customContextMenuRequested.connect(self.show_user_context_menu)
        layout.addWidget(self.content_label)
        self.main_frame.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def setup_ai_message(self, layout):
        """设置AI消息界面"""
        self.thinking_toggle = QPushButton("Thinking...")
        self.thinking_toggle.setObjectName("thinkingToggle")
        self.thinking_toggle.setFlat(True)
        self.thinking_toggle.clicked.connect(self.toggle_thinking)
        self.thinking_toggle.setCursor(Qt.CursorShape.PointingHandCursor)

        self.thinking_frame = QFrame()
        self.thinking_frame.setObjectName("thinkingFrame")
        self.thinking_frame.hide()
        thinking_frame_layout = QVBoxLayout(self.thinking_frame)
        thinking_frame_layout.setContentsMargins(0, 0, 0, 0)

        self.thinking_container = QWidget()
        self.thinking_layout = QVBoxLayout(self.thinking_container)
        self.thinking_layout.setContentsMargins(0, 0, 0, 0)
        self.thinking_layout.setSpacing(5)
        thinking_frame_layout.addWidget(self.thinking_container)

        self.answer_container = QWidget()
        self.answer_layout = QVBoxLayout(self.answer_container)
        self.answer_layout.setContentsMargins(0, 0, 0, 0)
        self.answer_layout.setSpacing(5)

        layout.addWidget(self.thinking_toggle)
        layout.addWidget(self.thinking_frame)
        layout.addWidget(self.answer_container)

        if not self.thinking:
            self.thinking_toggle.hide()

    def toggle_thinking(self):
        """切换思考过程显示状态"""
        self.thinking_visible = not self.thinking_visible
        self.thinking_frame.setVisible(self.thinking_visible)
        icon = "▼" if self.thinking_visible else "▶"
        text = "Thinking..." if self.is_streaming else "Thought"
        self.thinking_toggle.setText(f"{text} {icon}")

    def update_thinking(self, thinking_text: str):
        """更新思考过程"""
        self.thinking = thinking_text
        if not thinking_text:
            return

        self.thinking_toggle.show()
        if self.is_streaming and not self.thinking_visible:
            self.toggle_thinking()

        self._render_markdown_to_layout(thinking_text, self.thinking_layout)

    def update_answer(self, answer_text: str):
        """更新答案内容"""
        self.content = answer_text
        self._render_markdown_to_layout(self.content, self.answer_layout)

    def _clear_layout(self, layout):
        """清空布局中的所有小部件"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _render_markdown_to_layout(self, text: str, layout: QVBoxLayout):
        """解析Markdown文本并将其渲染到指定的布局中"""
        self._clear_layout(layout)

        if not text.strip():
            if layout is self.answer_layout:
                loading_label = QLabel("正在生成答案...")
                loading_label.setObjectName("loadingLabel")
                layout.addWidget(loading_label)
            return

        pattern = r"```(\w*)\n(.*?)```"
        parts = re.split(pattern, text, flags=re.DOTALL)

        for i, part in enumerate(parts):
            if not part:
                continue

            if i % 3 == 0:
                if part.strip():
                    html = self.md.convert(part)
                    if "<table>" in html:
                        html = self.get_table_styles() + html

                    text_label = QLabel(html)
                    text_label.setWordWrap(True)
                    text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    text_label.setObjectName("answerContent")
                    text_label.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                    text_label.customContextMenuRequested.connect(self.show_context_menu)
                    layout.addWidget(text_label)
            elif i % 3 == 1:
                language = part.strip() if part.strip() else "text"
                code = parts[i + 1]
                code_block = CodeBlockWidget(language, code)
                layout.addWidget(code_block)

        # 不使用stretch，保持高度随内容自适应
        if layout is self.answer_layout or layout is self.thinking_layout:
            layout.addSpacing(5)

    def update_content(self):
        """更新完整内容显示"""
        if self.role == "user":
            self.content_label.setText(self.content)
        else:
            if self.thinking:
                self.update_thinking(self.thinking)
                if not self.is_streaming:
                    self.thinking_toggle.setText("Thought ▶")
            self._render_markdown_to_layout(self.content, self.answer_layout)

    def show_user_context_menu(self, position):
        """显示用户消息的右键菜单"""
        menu = QMenu(self)
        menu.addAction("复制", self.copy_user_selection)

        widget = self.sender()
        if widget:
            menu.exec(widget.mapToGlobal(position))

    def copy_user_selection(self):
        """复制用户选中的文本，如果未选择则复制全部"""
        if self.content_label.hasSelectedText():
            QApplication.clipboard().setText(self.content_label.selectedText())
        else:
            QApplication.clipboard().setText(self.content)

    def show_context_menu(self, position):
        """显示右键上下文菜单"""
        menu = QMenu(self)
        if self.thinking:
            menu.addAction("复制思考过程", self.copy_thinking)
        if self.content:
            menu.addAction("复制答案", self.copy_answer)
        if self.content and not self.is_streaming:
            menu.addSeparator()
            menu.addAction("重新生成", self.regenerate_requested.emit)

        widget = self.sender()
        if widget and not menu.isEmpty():
            menu.exec(widget.mapToGlobal(position))

    def copy_thinking(self):
        QApplication.clipboard().setText(self.thinking)

    def copy_answer(self):
        QApplication.clipboard().setText(self.content)

    def on_thinking_complete(self):
        if self.thinking:
            if self.thinking_visible:
                self.toggle_thinking()
            self.thinking_toggle.setText("Thought ▶")

    def set_streaming_complete(self):
        self.is_streaming = False
