"""Instruction parsing and execution for the assistant chat.

Supports scanning directories and reading files under the project root with
basic path safety checks to prevent escaping the repository.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class InstructionExecutor:
    """指令执行器 - 处理SCAN_DIR, READ_FILE, ANALYZE三个指令"""

    def __init__(self, virgo_root: Optional[str] = None):
        # 自动检测Virgo根目录
        if virgo_root is None:
            current_dir = os.getcwd()
            if "Virgo" in current_dir:
                # 找到Virgo目录
                parts = current_dir.split(os.sep)
                virgo_index = None
                for i, part in enumerate(parts):
                    if "Virgo" in part:
                        virgo_index = i
                        break
                if virgo_index is not None:
                    self.virgo_root = os.sep.join(parts[: virgo_index + 1])
                else:
                    self.virgo_root = current_dir
            else:
                self.virgo_root = current_dir
        else:
            self.virgo_root = virgo_root

        self.virgo_root = os.path.realpath(self.virgo_root)
        logger.info("InstructionExecutor initialized with root: %s", self.virgo_root)

    def _resolve_under_root(self, target_path: Optional[str]) -> str:
        root = self.virgo_root
        if target_path is None:
            return root

        clean = str(target_path).strip().strip('"')
        if not clean:
            return root

        candidate = clean
        if not os.path.isabs(candidate):
            candidate = os.path.join(root, candidate)

        resolved = os.path.realpath(candidate)
        if resolved == root or resolved.startswith(root + os.sep):
            return resolved
        raise ValueError(f"路径必须位于 Virgo 根目录内: {clean}")

    def execute_scan_dir(self, target_path: Optional[str] = None) -> str:
        """扫描目录结构

        Args:
            target_path: 目标路径，如果为None则扫描根目录

        Returns:
            文件和目录路径列表，每行一个路径，用引号包围
        """
        is_root_scan = target_path is None or not str(target_path).strip().strip('"')
        try:
            scan_path = self._resolve_under_root(target_path)
        except ValueError as exc:
            return str(exc)

        if not os.path.exists(scan_path):
            return f"目录不存在: {scan_path}"

        file_list = []
        try:
            if is_root_scan:
                # 根目录扫描：只扫描第一层
                for item in sorted(os.listdir(scan_path)):
                    if item in {".git", "__pycache__", ".pytest_cache"}:
                        continue
                    item_path = os.path.join(scan_path, item)
                    file_list.append(f'"{item_path}"')
            else:
                # 子目录扫描：递归扫描所有内容
                for root, dirs, files in os.walk(scan_path):
                    dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache"}]
                    # 添加目录
                    level = root.replace(scan_path, "").count(os.sep)
                    if level >= 3:
                        dirs[:] = []
                    if level < 3:  # 限制层级，避免过深
                        for dir_name in sorted(dirs):
                            dir_path = os.path.join(root, dir_name)
                            file_list.append(f'"{dir_path}"')

                    # 添加文件
                    for file_name in sorted(files):
                        file_path = os.path.join(root, file_name)
                        file_list.append(f'"{file_path}"')

                    # 限制文件数量，避免输出过长
                    if len(file_list) > 100:
                        file_list.append('"...(更多文件,已截断)"')
                        break

        except Exception as e:
            return f"扫描目录出错: {str(e)}"

        return "\n".join(file_list[:50])  # 最多返回50个项目

    def execute_read_file(self, file_path: str) -> str:
        """读取文件内容

        Args:
            file_path: 文件路径，可能包含引号

        Returns:
            文件内容或错误信息
        """
        try:
            clean_path = self._resolve_under_root(file_path)
        except ValueError as exc:
            return str(exc)

        if not os.path.exists(clean_path):
            return f"文件不存在: {clean_path}"
        if not os.path.isfile(clean_path):
            return f"不是文件: {clean_path}"

        try:
            # 检查文件大小，避免读取过大文件
            file_size = os.path.getsize(clean_path)
            if file_size > 1024 * 1024:  # 1MB
                return f"文件过大 ({file_size} bytes)，跳过读取: {clean_path}"

            # 尝试UTF-8编码读取
            with open(clean_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 如果内容过长，截断
            if len(content) > 10000:  # 10K字符
                content = content[:10000] + "\n\n...(内容已截断，文件过长)"

            return content

        except UnicodeDecodeError:
            try:
                # 尝试GBK编码
                with open(clean_path, "r", encoding="gbk") as f:
                    content = f.read()
                if len(content) > 10000:
                    content = content[:10000] + "\n\n...(内容已截断，文件过长)"
                return content
            except Exception as e:
                return f"文件编码错误，无法读取: {str(e)}"
        except Exception as e:
            return f"读取文件出错: {str(e)}"


class InstructionParser:
    """指令解析器 - 从AI回复中提取指令"""

    @staticmethod
    def extract_instruction(response: str) -> Dict[str, Any]:
        """提取回复中的指令

        Args:
            response: AI回复文本

        Returns:
            包含指令类型和参数的字典
        """
        # 检查 [ANALYZE]
        if "[ANALYZE]" in response:
            return {"type": "ANALYZE", "param": None}

        # 检查 [SCAN_DIR:"path"]
        scan_match = re.search(r'\[SCAN_DIR:"([^"]+)"\]', response)
        if scan_match:
            return {"type": "SCAN_DIR", "param": scan_match.group(1)}

        # 检查 [SCAN_DIR] (无参数，扫描根目录)
        if "[SCAN_DIR]" in response:
            return {"type": "SCAN_DIR", "param": None}

        # 检查 [READ_FILE:"path"]
        read_match = re.search(r'\[READ_FILE:"([^"]+)"\]', response)
        if read_match:
            return {"type": "READ_FILE", "param": read_match.group(1)}

        return {"type": None, "param": None}


class StreamingInstructionController:
    """流式指令控制器 - 支持实时流式回复的指令处理"""

    def __init__(self, deepseek_client):
        self.deepseek_client = deepseek_client
        self.executor = InstructionExecutor()
        self.parser = InstructionParser()

        # 回调函数，用于向UI传递流式数据
        self.on_thinking_chunk = None
        self.on_answer_chunk = None
        self.on_thinking_complete = None
        self.on_iteration_start = None  # 新增：每轮开始回调
        self.on_iteration_complete = None  # 新增：每轮结束回调

    def set_streaming_callbacks(
        self,
        on_thinking_chunk,
        on_answer_chunk,
        on_thinking_complete,
        on_iteration_start=None,
        on_iteration_complete=None,
    ):
        """设置流式回调函数"""
        self.on_thinking_chunk = on_thinking_chunk
        self.on_answer_chunk = on_answer_chunk
        self.on_thinking_complete = on_thinking_complete
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete

    async def process_message_with_streaming(self, messages: List[Dict[str, str]]) -> str:
        """处理带指令的完整对话流程，支持流式回复

        Args:
            messages: 完整的对话历史，包含role和content

        Returns:
            最终的AI回复(包含[ANALYZE])
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        conversation_history: List[Dict[str, str]] = []

        for message in messages or []:
            role = message.get("role")
            content = message.get("content")

            if not role or content is None:
                continue

            conversation_history.append({"role": role, "content": content})

        if not conversation_history:
            logger.error("No valid conversation history provided for instruction processing")
            return "无法获取有效的对话内容，请重试。"
        max_iterations = 10  # 防止无限循环
        iteration_count = 0

        while iteration_count < max_iterations:
            iteration_count += 1

            try:
                # 通知UI新的迭代开始
                if self.on_iteration_start:
                    self.on_iteration_start(iteration_count)

                # 获取AI回复（流式）
                thinking_content, ai_response = await self._get_ai_response_streaming(
                    conversation_history, iteration_count
                )

                # 提取指令
                instruction = self.parser.extract_instruction(ai_response)

                logger.info(
                    f"Iteration {iteration_count}: Found instruction type: {instruction['type']}"
                )

                if instruction["type"] == "ANALYZE":
                    # 最终回复，返回给用户
                    return ai_response

                # 在执行指令前，调用迭代完成回调
                if self.on_iteration_complete:
                    self.on_iteration_complete(iteration_count, thinking_content, ai_response)

                if instruction["type"] == "SCAN_DIR":
                    # 执行目录扫描
                    result = self.executor.execute_scan_dir(instruction["param"])

                    # 构建新的对话继续
                    conversation_history.extend(
                        [
                            {"role": "assistant", "content": ai_response},
                            {"role": "user", "content": f"# Result:\n{result}"},
                        ]
                    )

                elif instruction["type"] == "READ_FILE":
                    # 执行文件读取
                    result = self.executor.execute_read_file(instruction["param"])

                    # 构建新的对话继续
                    conversation_history.extend(
                        [
                            {"role": "assistant", "content": ai_response},
                            {"role": "user", "content": f"# Result:\n{result}"},
                        ]
                    )

                else:
                    # 没有找到有效指令，可能是普通回复
                    logger.warning(
                        f"No valid instruction found in response: {ai_response[:100]}..."
                    )
                    return ai_response

            except Exception as e:
                logger.error(
                    f"Error in instruction processing iteration {iteration_count}: {str(e)}"
                )
                return f"处理指令时发生错误: {str(e)}"

        # 超过最大迭代次数
        return "指令处理超时，请重新提问。"

    async def _get_ai_response_streaming(
        self, messages: List[Dict[str, str]], iteration: int
    ) -> Tuple[str, str]:
        """获取AI回复（流式）

        Args:
            messages: 对话历史
            iteration: 当前迭代次数

        Returns:
            (thinking内容, answer内容)
        """
        # 重置客户端状态
        self.deepseek_client._current_thinking = ""
        self.deepseek_client._current_answer = ""
        self.deepseek_client._in_thinking = False
        self.deepseek_client._in_answer = False

        # 重置去重跟踪变量
        self.deepseek_client._last_emitted_thinking = ""
        self.deepseek_client._last_emitted_answer = ""

        # 强制断开所有现有连接，防止多重连接
        try:
            self.deepseek_client.thinking_chunk.disconnect()
            self.deepseek_client.answer_chunk.disconnect()
            self.deepseek_client.thinking_complete.disconnect()
        except (TypeError, RuntimeError):
            # 忽略断开连接时的错误
            pass

        # 创建带迭代信息的回调包装器
        def wrapped_thinking_chunk(chunk):
            if self.on_thinking_chunk:
                self.on_thinking_chunk(chunk, iteration)

        def wrapped_answer_chunk(chunk):
            if self.on_answer_chunk:
                self.on_answer_chunk(chunk, iteration)

        def wrapped_thinking_complete():
            if self.on_thinking_complete:
                self.on_thinking_complete(iteration)

        try:
            # 连接信号 - 使用包装器，确保单一连接
            self.deepseek_client.thinking_chunk.connect(wrapped_thinking_chunk)
            self.deepseek_client.answer_chunk.connect(wrapped_answer_chunk)
            self.deepseek_client.thinking_complete.connect(wrapped_thinking_complete)

            # 调用DeepSeek客户端获取流式回复
            await self.deepseek_client.chat_stream(messages)

            # 返回完整的thinking和answer内容
            return self.deepseek_client._current_thinking, self.deepseek_client._current_answer

        except Exception as e:
            logger.error(f"获取AI回复失败: {str(e)}")
            return self.deepseek_client._current_thinking, f"获取AI回复时发生错误: {str(e)}"
        finally:
            # 强制断开所有信号连接
            try:
                self.deepseek_client.thinking_chunk.disconnect()
                self.deepseek_client.answer_chunk.disconnect()
                self.deepseek_client.thinking_complete.disconnect()
            except (TypeError, RuntimeError):
                # 忽略断开连接时的错误
                pass


# 保持向后兼容
InstructionController = StreamingInstructionController

# 移除should_analyze_project方法 - 现在完全由AI在prompt中判断
