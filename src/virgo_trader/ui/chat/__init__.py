"""
AI疑难解答聊天模块

支持DeepSeek R1的思考过程显示和聊天历史管理
"""

from .chat_widget import ChatWidget
from .history_sidebar import HistorySidebar
from .message_bubble import MessageBubble

__all__ = ["ChatWidget", "MessageBubble", "HistorySidebar"]
