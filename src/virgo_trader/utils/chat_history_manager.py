"""Persist chat conversation history for the in-app assistant.

Conversations are stored in a JSON file and exposed via simple CRUD helpers used
by the dashboard chat UI.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .paths import CHAT_HISTORY_PATH, migrate_legacy_files

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """聊天历史管理器"""

    def __init__(self, history_file: Optional[str] = None):
        migrate_legacy_files()
        self.history_file = str(CHAT_HISTORY_PATH) if history_file is None else history_file
        self.conversations = {}
        self.load_history()

    def load_history(self):
        """从文件加载聊天历史"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 兼容旧格式
                if isinstance(data, list):
                    # 旧格式：直接是消息列表，转换为新格式
                    conversation_id = str(uuid4())
                    self.conversations = {
                        conversation_id: {
                            "id": conversation_id,
                            "title": "历史对话",
                            "created_at": datetime.now().isoformat(),
                            "updated_at": datetime.now().isoformat(),
                            "messages": data,
                        }
                    }
                else:
                    # 新格式：对话字典
                    self.conversations = data

        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("加载聊天历史失败: %s", e)
            self.conversations = {}

    def save_history(self):
        """保存聊天历史到文件"""
        try:
            dir_path = os.path.dirname(self.history_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning("保存聊天历史失败: %s", e)

    def create_conversation(self, title: Optional[str] = None) -> str:
        """创建新对话"""
        conversation_id = str(uuid4())
        now = datetime.now().isoformat()

        self.conversations[conversation_id] = {
            "id": conversation_id,
            "title": title or "新对话",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

        self.save_history()
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取对话"""
        return self.conversations.get(conversation_id)

    def get_conversations_by_date(self) -> Dict[str, List[Dict[str, Any]]]:
        """按日期分组获取对话列表"""
        grouped = {"今天": [], "昨天": [], "本周": [], "更早": []}

        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        week_ago = today - timedelta(days=7)

        for conv in self.conversations.values():
            try:
                created_date = datetime.fromisoformat(conv["created_at"]).date()

                if created_date == today:
                    grouped["今天"].append(conv)
                elif created_date == yesterday:
                    grouped["昨天"].append(conv)
                elif created_date > week_ago:
                    grouped["本周"].append(conv)
                else:
                    grouped["更早"].append(conv)

            except (ValueError, KeyError):
                # 处理日期解析错误
                grouped["更早"].append(conv)

        # 按更新时间排序
        for group in grouped.values():
            group.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return grouped

    def add_message(
        self, conversation_id: str, role: str, content: str, thinking: Optional[str] = None
    ) -> bool:
        """添加消息到对话"""
        if conversation_id not in self.conversations:
            return False

        message = {"role": role, "content": content, "timestamp": datetime.now().isoformat()}

        # 如果是AI回复且有思考过程，添加thinking字段
        if thinking and role == "assistant":
            message["thinking"] = thinking

        self.conversations[conversation_id]["messages"].append(message)
        self.conversations[conversation_id]["updated_at"] = datetime.now().isoformat()

        # 自动更新对话标题
        self._update_conversation_title(conversation_id)

        self.save_history()
        return True

    def _update_conversation_title(self, conversation_id: str):
        """自动更新对话标题为第一个用户消息的前20个字符"""
        conv = self.conversations[conversation_id]

        # 如果标题是默认的"新对话"，尝试更新
        if conv.get("title") == "新对话":
            for msg in conv["messages"]:
                if msg["role"] == "user" and msg["content"].strip():
                    # 取前20个字符作为标题
                    title = msg["content"].strip()[:20]
                    if len(msg["content"]) > 20:
                        title += "..."
                    conv["title"] = title
                    break

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除对话"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.save_history()
            return True
        return False

    def clear_all_conversations(self):
        """清空所有对话"""
        self.conversations = {}
        self.save_history()

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """获取对话的所有消息"""
        conv = self.get_conversation(conversation_id)
        return conv["messages"] if conv else []

    def update_conversation_title(self, conversation_id: str, new_title: str) -> bool:
        """更新对话标题"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["title"] = new_title
            self.conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
            self.save_history()
            return True
        return False
