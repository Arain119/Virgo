"""Conversation history sidebar widget for the chat UI.

Provides grouping, selection, and basic conversation management actions.
"""

import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...utils.chat_history_manager import ChatHistoryManager
from ..styled_message_box import StyledMessageBox

logger = logging.getLogger(__name__)


class ConversationItem(QPushButton):
    """对话项目组件"""

    clicked_with_id = pyqtSignal(str)  # conversation_id
    delete_requested = pyqtSignal(str)  # conversation_id

    def __init__(self, conversation_data, parent=None):
        super().__init__(parent)
        self.conversation_data = conversation_data
        self.conversation_id = conversation_data["id"]
        self.is_selected = False

        self.setup_ui()

        # 连接点击信号
        self.clicked.connect(self.on_clicked)

    def setup_ui(self):
        """设置用户界面"""
        # 设置按钮样式为左对齐，移除边框和背景，减少左padding让文字更靠左
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px 12px 10px 5px;
                color: #f8f8f2;
                font-size: 13px;
                font-weight: bold;
                border: none;
                background: transparent;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: rgba(189, 147, 249, 0.1);
            }
            QPushButton:pressed {
                background-color: rgba(189, 147, 249, 0.2);
            }
        """)

        # 设置按钮文本，只显示标题，不显示时间
        title = self.conversation_data["title"]
        self.setText(title)

        # 设置右键菜单
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def on_clicked(self):
        """处理按钮点击"""
        self.clicked_with_id.emit(self.conversation_id)

    def _format_time(self):
        """格式化时间显示"""
        from datetime import datetime

        try:
            created_time = datetime.fromisoformat(self.conversation_data["created_at"])
            return created_time.strftime("%m-%d %H:%M")
        except (KeyError, ValueError, TypeError):
            return ""

    def mousePressEvent(self, event):
        """鼠标点击事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            logger.debug("ConversationItem mousePressEvent: %s", self.conversation_id)
            # 让默认的按钮处理机制工作，不要在这里重复发射信号
        super().mousePressEvent(event)

    def show_context_menu(self, position):
        """显示右键菜单"""
        menu = QMenu(self)

        rename_action = QAction("重命名", self)
        rename_action.triggered.connect(self.rename_conversation)
        menu.addAction(rename_action)

        delete_action = QAction("删除", self)
        delete_action.triggered.connect(self.delete_conversation)
        menu.addAction(delete_action)

        menu.exec(self.mapToGlobal(position))

    def rename_conversation(self):
        """重命名对话"""
        new_title, ok = QInputDialog.getText(
            self, "重命名对话", "请输入新标题:", text=self.conversation_data["title"]
        )

        if ok and new_title.strip():
            # 通知父组件处理重命名
            if hasattr(self.parent(), "rename_conversation"):
                self.parent().rename_conversation(self.conversation_id, new_title.strip())

    def delete_conversation(self):
        """删除对话"""
        if StyledMessageBox.question(
            self, "确认删除", f'确定要删除对话 "{self.conversation_data["title"]}" 吗？'
        ):
            self.delete_requested.emit(self.conversation_id)

    def set_selected(self, selected):
        """设置选中状态"""
        self.is_selected = selected
        if selected:
            self.setObjectName("conversationItemSelected")
        else:
            self.setObjectName("conversationItem")
        self.style().polish(self)  # 重新应用样式


class HistorySidebar(QWidget):
    """历史记录侧栏"""

    conversation_selected = pyqtSignal(str)  # conversation_id
    new_conversation_requested = pyqtSignal()
    conversation_deleted = pyqtSignal(str)  # conversation_id

    def __init__(self, history_manager: ChatHistoryManager, parent=None):
        super().__init__(parent)
        self.history_manager = history_manager
        self.selected_conversation_id = None
        self.conversation_items = {}

        self.setup_ui()
        self.refresh_conversations()

    def setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 整体容器框架 - 包含标题和历史记录
        header = QFrame()
        header.setObjectName("chatSidebarHeader")
        header_layout = QVBoxLayout(header)  # 改为垂直布局
        header_layout.setContentsMargins(20, 15, 15, 15)  # 标题向右下移动一点
        header_layout.setSpacing(2)  # 进一步减少标题和列表之间的间距，让列表更靠近标题

        # 顶部标题行
        title_row = QHBoxLayout()
        title = QLabel("历史会话")
        title.setObjectName("chatSidebarTitle")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)

        self.new_chat_btn = QPushButton("+")
        self.new_chat_btn.setObjectName("newChatButton")
        self.new_chat_btn.clicked.connect(self.new_conversation_requested.emit)

        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.new_chat_btn)

        header_layout.addLayout(title_row)

        # 对话列表滚动区域 - 现在在header内部
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setObjectName("chatHistoryScrollArea")
        self.scroll_area.setMinimumHeight(300)  # 设置最小高度

        # 对话列表容器
        self.conversations_widget = QWidget()
        self.conversations_layout = QVBoxLayout(self.conversations_widget)
        self.conversations_layout.setContentsMargins(0, 3, 3, 3)
        self.conversations_layout.setSpacing(1)
        self.conversations_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 顶部对齐，不居中
        self.conversations_layout.addStretch()  # 将内容推到顶部

        self.scroll_area.setWidget(self.conversations_widget)
        header_layout.addWidget(self.scroll_area)

        layout.addWidget(header)

    def refresh_conversations(self):
        """刷新对话列表"""
        # 清空现有items
        self.clear_conversations()

        # 按日期分组获取对话
        grouped_conversations = self.history_manager.get_conversations_by_date()

        for group_name, conversations in grouped_conversations.items():
            if not conversations:
                continue

            # 添加分组标题
            if group_name != "今天" or len(conversations) > 0:
                self.add_group_header(group_name)

            # 添加对话项目
            for conv in conversations:
                self.add_conversation_item(conv)

    def clear_conversations(self):
        """清空对话列表"""
        # 移除除了stretch之外的所有widget
        while self.conversations_layout.count() > 1:
            item = self.conversations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.conversation_items.clear()

    def add_group_header(self, group_name):
        """添加分组标题"""
        header = QLabel(group_name)
        header.setObjectName("conversationGroupHeader")
        header.setContentsMargins(5, 10, 15, 3)

        font = header.font()
        font.setBold(True)
        font.setPointSize(6)
        header.setFont(font)

        # 在stretch之前插入
        count = self.conversations_layout.count()
        self.conversations_layout.insertWidget(count - 1, header)

    def add_conversation_item(self, conversation_data):
        """添加对话项目"""
        item = ConversationItem(conversation_data, self)
        item.setObjectName("conversationItem")
        # Connect to the custom signal that carries the conversation id.
        item.clicked_with_id.connect(self.on_conversation_clicked)
        item.delete_requested.connect(self.on_conversation_delete_requested)

        self.conversation_items[conversation_data["id"]] = item

        # 在stretch之前插入
        count = self.conversations_layout.count()
        self.conversations_layout.insertWidget(count - 1, item)

    def on_conversation_clicked(self, conversation_id):
        """对话被点击"""
        logger.debug("HistorySidebar.on_conversation_clicked: %s", conversation_id)
        self.select_conversation(conversation_id)
        self.conversation_selected.emit(conversation_id)

    def select_conversation(self, conversation_id):
        """选中对话"""
        # 取消之前的选中状态
        if self.selected_conversation_id in self.conversation_items:
            self.conversation_items[self.selected_conversation_id].set_selected(False)

        # 设置新的选中状态
        self.selected_conversation_id = conversation_id
        if conversation_id in self.conversation_items:
            self.conversation_items[conversation_id].set_selected(True)

    def on_conversation_delete_requested(self, conversation_id):
        """对话删除请求"""
        # 从历史管理器中删除
        if self.history_manager.delete_conversation(conversation_id):
            # 刷新显示
            self.refresh_conversations()
            # 通知外部
            self.conversation_deleted.emit(conversation_id)

    def rename_conversation(self, conversation_id, new_title):
        """重命名对话"""
        if self.history_manager.update_conversation_title(conversation_id, new_title):
            # 刷新显示
            self.refresh_conversations()
            # 重新选中该对话
            if conversation_id == self.selected_conversation_id:
                self.select_conversation(conversation_id)
