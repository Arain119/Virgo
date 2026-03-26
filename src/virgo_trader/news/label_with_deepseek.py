"""LLM labeling pipeline using DeepSeek (via SiliconFlow API).

Reads exported CSV datasets, calls the model to label sentiment/emotions, and
writes JSONL outputs (plus an error JSONL for failed records).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import requests

from . import config
from .label_with_fomalhaut import (
    ModelResponseError,
    build_prompt,
    parse_model_response,
    sanitize_range_label,
)

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
DEFAULT_OUTPUT = config.EXPORT_DIR / "news_deepseek_labels.jsonl"
DEFAULT_ERROR_OUTPUT = config.EXPORT_DIR / "news_deepseek_errors.jsonl"
DEFAULT_SPLIT_DIR = config.EXPORT_DIR
SYSTEM_PROMPT = """**角色:**
你是一位具备深度语言理解、世界知识和常识推理能力的专家级情感分析师。你的核心任务是评估给定文本内容对其 **典型核心受众** 可能产生的 **情感冲击力**，并且分析表层语言情感。你的分析需要体现 **内容驱动** 和 **受众感知** 的原则。你的输出将用于知识蒸馏，必须是精确、结构化且格式严格一致的 JSON 字符串。

**任务:**
仔细阅读并深刻理解下面提供的 "[输入文本]"。运用你的世界知识和常识推理能力：
1.  **推断核心受众:** 基于文本内容、类型（如新闻、评论、公告）和潜在来源，推断该信息主要影响的典型受众群体（例如：投资者、普通消费者、特定社群成员、一般公众等）。
2.  **模拟认知评价:** 模拟该典型受众在接收此信息后，可能进行的认知评价过程（思考事件的相关性、对目标的潜在影响、应对可能性、责任归属等）。
3.  **评估典型情感冲击力:** 基于上述评价，识别该内容最可能引发的 **一个或多个** 典型情感冲击反应，并评估其冲击程度或事件严重性。
4.  **应用内容优先规则:** 当文本的表层语言情感（如积极/消极词汇）与其基于内容理解、常识推理和受众模拟推断出的典型情感冲击力发生 **冲突** 时，**必须优先采纳后者（内容冲击力）**。例如，一则语言中性的“大股东减持”公告，对投资者受众的典型冲击力应判断为负面（如焦虑）。一则措辞礼貌的“服务中断”通知，对用户的冲击力应判断为轻微负面（如无奈、惊讶）。
5.  **构造输出:** 将你的分析结果构造为一个单一的、符合规范的 JSON 字符串，并将其作为唯一的输出。

**情感冲击力框架定义**

1.  **情感冲击力标签（必须从以下列表中选择，可多选）:**
    *   快乐
    *   悲伤
    *   愤怒
    *   恐惧
    *   惊讶
    *   厌恶
    *   愧疚
    *   希望
    *   焦虑
    *   信任
    *   满意
    *   失望
    *   羞愧
    *   轻蔑
    *   同情
    *   委屈
    *   无奈
    *   纠结
    *   心疼
    *   幸灾乐祸
    *   尴尬
    *   自豪
    *   中性

2.  **情感冲击力强度（为每个识别出的标签分配一个 1 到 7 的整数评分，反映冲击程度或事件严重性）:**
    *   1: 微乎其微/几乎无感的冲击 / 事件几乎无足轻重
    *   2: 冲击力极轻微 / 事件影响极小
    *   3: 冲击力轻微 / 事件影响较小
    *   4: 冲击力中等 / 事件影响一般
    *   5: 冲击力强烈 / 事件影响显著
    *   6: 冲击力非常强烈 / 事件影响重大
    *   7: 冲击力极其强烈 / 压倒性 / 事件影响极端或灾难级

**输出格式要求:**

你 **必须** 将结果输出为一个 **单一的、位于一行内的 JSON 字符串**。
JSON 对象包含 `"labels"` 和 `"intensities"` 两个键。

*   `"labels"`: 一个 JSON 数组，包含所有识别出的情感冲击力标签字符串。
*   `"intensities"`: 一个 JSON 数组，包含对应于 `"labels"` 数组中每个标签的强度整数（1-7）。顺序必须严格一一对应。

**极其重要:**
*   你的最终输出 **必须且只能是** `{ "labels": [...], "intensities": [...] }` 格式的 JSON 字符串，**禁止** 任何解释、代码标记、问候语等。
*   **多标签识别** 务必识别并包含所有适用的情感冲击标签及其强度。
*   **中性处理**
    *   若内容对典型核心受众 **确实没有** 可预测的情感冲击力（仅为客观信息传递），则 **必须** 输出且仅输出: `{"labels": ["中性"], "intensities": [1]}`。强度固定为 1。
    *   只要识别出任何一个或多个具体的情感冲击 (非“中性”)，则 **绝对不应该** 在 `"labels"` 数组中包含“中性”。
*   **内容优先规则的应用** 时刻记住，基于内容的深层冲击力判断优先于表面语言情感。

**请严格按照上述指示，分析以下文本，评估其对典型核心受众的情感冲击力，并仅输出符合要求的单一 JSON 字符串。**
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 SiliconFlow deepseek 模型顺序标注新闻（system/prompt 与本地流程保持一致）"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="需要处理的 CSV 文件路径，可重复多次（按提供顺序处理）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"情感结果输出（JSONL，默认 {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=DEFAULT_ERROR_OUTPUT,
        help=f"失败记录输出（JSONL，默认 {DEFAULT_ERROR_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SiliconFlow 模型名称（默认 {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"SiliconFlow Chat Completions API (默认 {API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SILICONFLOW_API_KEY", None),
        help="API Key（默认读取 SILICONFLOW_API_KEY 环境变量，也可用 --api-key 显式指定）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="模型调用失败时的最大重试次数（默认 5，最多 5 次）",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="每条请求之间的休眠秒数（默认 0.5）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 条记录（0 表示全部）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在，重新开始并覆盖；默认在原文件基础上断点续跑",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志等级（默认 INFO）",
    )
    parser.add_argument(
        "--split-output-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help=f"按 range_label 拆分输出 JSONL 的目录（默认 {DEFAULT_SPLIT_DIR}）",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="禁用按 range_label 拆分输出，仅写入 --output",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="并发 worker 数量（默认 4，需自行确保符合 RPM/TPM 限制）",
    )
    return parser.parse_args()


def load_processed_hashes(path: Path, overwrite: bool) -> Set[str]:
    processed: Set[str] = set()
    if overwrite or not path.exists():
        if overwrite and path.exists():
            path.unlink()
        return processed
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_hash = payload.get("hash")
            if record_hash:
                processed.add(record_hash)
    logging.info("已加载 %s 条历史记录，将在此基础上续跑", len(processed))
    return processed


def ensure_path(path: Path, overwrite: bool = False) -> None:
    if overwrite and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_split_outputs(
    split_dir: Optional[Path], overwrite: bool, aggregate_path: Path, error_path: Path
) -> None:
    if split_dir is None:
        return
    split_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    aggregate_resolved = aggregate_path.resolve()
    error_resolved = error_path.resolve()
    for file in split_dir.glob("news_deepseek_*.jsonl"):
        resolved = file.resolve()
        if resolved in (aggregate_resolved, error_resolved):
            continue
        try:
            file.unlink()
        except FileNotFoundError:
            continue


def get_split_output_path(split_dir: Path, range_label: Optional[str]) -> Path:
    sanitized = sanitize_range_label(range_label)
    return split_dir / f"news_deepseek_{sanitized}.jsonl"


def write_jsonl(path: Path, payload: Dict[str, object], lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def call_model(
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_retries: int,
) -> Tuple[Dict[str, object], str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "enable_thinking": True,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    attempts = max(1, min(max_retries, 5))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        session = requests.Session()
        try:
            resp = session.post(api_url, headers=headers, json=payload, timeout=240)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            content = None
            for choice in choices:
                message = choice.get("message") or {}
                text = message.get("content")
                if text:
                    content = text
                    break
            if not content:
                serialized = json.dumps(data, ensure_ascii=False)
                raise ModelResponseError("模型响应缺少内容", raw_response=serialized)
            analysis = parse_model_response(content)
            return analysis, content
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            logging.warning("SiliconFlow 调用失败（第 %s/%s 次）: %s", attempt, attempts, exc)
            time.sleep(min(2.0 * attempt, 5.0))
        finally:
            session.close()
    if last_error is None:
        raise RuntimeError("SiliconFlow 调用失败，但未捕获到异常信息")
    raise last_error


def process_file(
    csv_path: Path,
    processed: Set[str],
    args: argparse.Namespace,
    output_path: Path,
    error_path: Path,
    split_dir: Optional[Path],
    processed_lock: threading.Lock,
    write_locks: Dict[Path, threading.Lock],
) -> Tuple[int, int]:
    logging.info("开始处理 %s", csv_path)
    total = 0
    success = 0
    limit = args.limit
    aggregate_resolved = output_path.resolve()
    inflight: Set[str] = set()
    futures: Dict[Future, Tuple[str, Dict[str, str], str]] = {}

    def get_lock(path: Path) -> threading.Lock:
        lock = write_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            write_locks[path] = lock
        return lock

    def submit_task(row: Dict[str, str]) -> Optional[Future]:
        record_hash = row.get("hash")
        if not record_hash:
            logging.warning("第 %s 条缺少 hash，已跳过", total)
            return None
        with processed_lock:
            if record_hash in processed or record_hash in inflight:
                return None
            inflight.add(record_hash)
        prompt = build_prompt(row)

        def worker(row_data: Dict[str, str], prompt_text: str) -> Tuple[str, Dict[str, object]]:
            record_hash_inner = row_data.get("hash")
            try:
                analysis, raw_text = call_model(
                    api_url=args.api_url,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=prompt_text,
                    max_retries=args.max_retries,
                )
                payload: Dict[str, object] = {
                    "hash": record_hash_inner,
                    "symbol": row_data.get("symbol"),
                    "security_name": row_data.get("security_name"),
                    "range_label": row_data.get("range_label"),
                    "source": row_data.get("source"),
                    "news_type": row_data.get("news_type"),
                    "published_at": row_data.get("published_at"),
                    "labels": analysis.get("labels"),
                    "intensities": analysis.get("intensities"),
                    "model": args.model,
                    "prompt": prompt_text,
                    "raw_response": raw_text,
                }
                return "ok", payload
            except Exception as exc:  # noqa: PERF203
                error_payload = {
                    "hash": record_hash_inner,
                    "symbol": row_data.get("symbol"),
                    "range_label": row_data.get("range_label"),
                    "source_file": str(csv_path),
                    "error": str(exc),
                    "prompt": prompt_text,
                }
                if isinstance(exc, ModelResponseError) and getattr(exc, "raw_response", None):
                    error_payload["raw_response"] = exc.raw_response
                return "error", error_payload
            finally:
                time.sleep(args.sleep)

        future = executor.submit(worker, row.copy(), prompt)
        futures[future] = (record_hash, row, prompt)
        return future

    max_workers = max(1, args.concurrency)
    with (
        csv_path.open("r", encoding="utf-8-sig", newline="") as fh,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        import csv

        reader = csv.DictReader(fh)
        submitted = 0
        for row in reader:
            total += 1
            if limit and submitted >= limit:
                break
            fut = submit_task(row)
            if fut:
                submitted += 1

        for future in as_completed(list(futures.keys())):
            record_hash, row_data, _prompt = futures[future]
            status, payload = future.result()
            with processed_lock:
                inflight.discard(record_hash)
            if status == "ok":
                write_jsonl(output_path, payload, get_lock(output_path))
                if split_dir is not None:
                    split_path = get_split_output_path(split_dir, row_data.get("range_label"))
                    if split_path.resolve() != aggregate_resolved:
                        write_jsonl(split_path, payload, get_lock(split_path))
                with processed_lock:
                    processed.add(record_hash)
                success += 1
            else:
                write_jsonl(error_path, payload, get_lock(error_path))

    logging.info("文件 %s 处理完成：成功 %s / 遍历 %s", csv_path, success, total)
    return total, success


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if not args.api_key:
        raise SystemExit("必须提供 API Key (参数 --api-key 或环境变量 SILICONFLOW_API_KEY)")
    input_paths = [Path(p) for p in args.input]
    split_dir: Optional[Path] = None
    if not args.no_split:
        split_dir = args.split_output_dir
    ensure_path(args.output, overwrite=args.overwrite)
    ensure_path(args.error_output, overwrite=args.overwrite)
    prepare_split_outputs(split_dir, args.overwrite, args.output, args.error_output)
    processed = load_processed_hashes(args.output, args.overwrite)
    total_entries = 0
    total_success = 0
    processed_lock = threading.Lock()
    write_locks: Dict[Path, threading.Lock] = {}
    try:
        for path in input_paths:
            file_total, file_success = process_file(
                csv_path=path,
                processed=processed,
                args=args,
                output_path=args.output,
                error_path=args.error_output,
                split_dir=split_dir,
                processed_lock=processed_lock,
                write_locks=write_locks,
            )
            total_entries += file_total
            total_success += file_success
            if args.limit and total_success >= args.limit:
                break
    except KeyboardInterrupt:
        logging.warning("用户中断，已保存当前进度（成功 %s 条）", total_success)
    logging.info(
        "全部文件处理完毕：累计成功 %s 条 / 遍历 %s 条，结果写入 %s",
        total_success,
        total_entries,
        args.output,
    )


if __name__ == "__main__":
    main()
