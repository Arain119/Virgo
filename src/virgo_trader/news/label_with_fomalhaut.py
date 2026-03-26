"""LLM labeling pipeline using Fomalhaut models.

Reads exported CSV datasets, invokes the model to label sentiment/emotions, and
writes JSONL outputs used by downstream sentiment feature generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests

from . import config

DEFAULT_MODEL = "Fomalhaut:latest"
DEFAULT_HOST = "http://127.0.0.1:11434"
DEFAULT_OUTPUT = config.EXPORT_DIR / "news_fomalhaut_labels.jsonl"
DEFAULT_ERROR_OUTPUT = config.EXPORT_DIR / "news_fomalhaut_errors.jsonl"
DEFAULT_SPLIT_DIR = config.EXPORT_DIR


class ModelResponseError(RuntimeError):
    def __init__(self, message: str, raw_response: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Fomalhaut 模型逐条分析新闻 CSV 并写入情感数据集（顺序处理，无并发）"
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
        help=f"调用的 Ollama 模型名称（默认 {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ollama 服务地址（默认 {DEFAULT_HOST})",
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
        help="禁用按 range_label 拆分输出，仅生成 --output 与 --error-output",
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


def ensure_error_output(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output(path: Path) -> None:
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
    for file in split_dir.glob("news_fomalhaut_*.jsonl"):
        resolved = file.resolve()
        if resolved == aggregate_resolved or resolved == error_resolved:
            continue
        try:
            file.unlink()
        except FileNotFoundError:
            continue


def build_prompt(row: Dict[str, str]) -> str:
    summary = (row.get("summary") or "").strip()
    if not summary:
        summary = "（无摘要）"
    headline = (row.get("headline") or "").strip()
    range_label = (row.get("range_label") or "").strip() or "unknown_range"
    lines = [
        f"标的: {row.get('security_name', '未知标的')} ({row.get('symbol', 'N/A')})",
        f"来源: {row.get('source', 'unknown')} / {row.get('news_type', '-')} / {range_label}",
        f"标题: {headline or '（无标题）'}",
        f"摘要: {summary}",
    ]
    body = "\n".join(lines)
    prompt = (
        "下面是需要分析的输入文本：\n"
        "---\n"
        f"{body}\n"
        "---\n"
        "请严格依照系统设定的格式要求，仅返回 JSON 结果。"
    )
    return prompt


def _ensure_text_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_intensity(value: object, treat_as_score: bool = False) -> int:
    num = _coerce_float(value)
    if num is None:
        return 4
    if treat_as_score:
        num = num * 6 + 1  # scale 0-1 score to 1-7 range
    return max(1, min(7, int(round(num))))


def _align_intensities(values: List[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return values
    if not values:
        return [4] * target_len
    if len(values) == target_len:
        return values
    if len(values) == 1:
        return values * target_len
    trimmed = values[:target_len]
    while len(trimmed) < target_len:
        trimmed.append(values[-1])
    return trimmed


def _parse_candidate(candidate: Dict[str, object]) -> Optional[Dict[str, List[int]]]:
    labels = None
    intensities: List[int] = []
    if "labels" in candidate:
        labels = _ensure_text_list(candidate.get("labels"))
        intensities = _align_intensities(
            [_coerce_intensity(item) for item in _ensure_text_list(candidate.get("intensities"))],
            len(labels),
        )
        if not any(intensities) and candidate.get("scores") is not None:
            intensities = _align_intensities(
                [
                    _coerce_intensity(item, treat_as_score=True)
                    for item in _ensure_text_list(candidate.get("scores"))
                ],
                len(labels),
            )
    if not labels:
        for key in ("label", "emotion", "feeling", "sentiment"):
            if key in candidate:
                labels = _ensure_text_list(candidate.get(key))
                break
    if labels is not None and not intensities:
        source = None
        treat_as_score = False
        for key in ("intensities", "intensity", "score", "scores", "strength", "strengths"):
            if key in candidate:
                source = candidate.get(key)
                treat_as_score = key in {"score", "scores"}
                break
        if source is not None:
            intensities = _align_intensities(
                [
                    _coerce_intensity(item, treat_as_score=treat_as_score)
                    for item in _ensure_text_list(source)
                ],
                len(labels),
            )
    if labels:
        if not intensities:
            intensities = [4] * len(labels)
        else:
            intensities = _align_intensities(intensities, len(labels))
        return {"labels": labels, "intensities": intensities}
    results = candidate.get("results")
    if isinstance(results, list):
        for entry in results:
            if isinstance(entry, dict):
                parsed = _parse_candidate(entry)
                if parsed:
                    return parsed
    return None


def parse_model_response(text: str) -> Dict[str, List[int]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ModelResponseError(f"无法解析模型响应: {exc}", raw_response=text) from exc
    candidate: Optional[Dict[str, object]] = None
    if isinstance(payload, dict):
        candidate = payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                candidate = item
                break
    if not candidate:
        raise ModelResponseError("模型响应缺少可解析的对象", raw_response=text)
    parsed = _parse_candidate(candidate)
    if not parsed or not parsed.get("labels"):
        raise ModelResponseError("模型响应未包含可解析的情感标签", raw_response=text)
    labels = [label.strip() for label in parsed["labels"] if label and label.strip()]
    if not labels:
        raise ModelResponseError("模型响应提供的标签为空", raw_response=text)
    intensities = parsed.get("intensities") or [4] * len(labels)
    intensities = _align_intensities(intensities, len(labels))
    return {"labels": labels, "intensities": intensities}


def sanitize_range_label(label: Optional[str]) -> str:
    if not label:
        return "unknown"
    cleaned_chars: List[str] = []
    for ch in label:
        if ch.isalnum() or ch in {"_", "-"}:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append("_")
    sanitized = "".join(cleaned_chars).strip("_")
    return sanitized or "unknown"


def get_split_output_path(split_dir: Path, range_label: Optional[str]) -> Path:
    sanitized = sanitize_range_label(range_label)
    return split_dir / f"news_fomalhaut_{sanitized}.jsonl"


def call_model(
    session: requests.Session,
    host: str,
    model: str,
    prompt: str,
    max_retries: int,
) -> Tuple[Dict[str, object], str]:
    url = host.rstrip("/") + "/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
    last_error: Exception | None = None
    attempts = max(1, min(max_retries, 5))
    for attempt in range(1, attempts + 1):
        try:
            resp = session.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("response") or "").strip()
            analysis = parse_model_response(text)
            return analysis, text
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            logging.warning("模型调用失败（第 %s/%s 次）: %s", attempt, attempts, exc)
            time.sleep(min(2.0 * attempt, 5.0))
    if last_error is None:
        raise RuntimeError("模型调用失败，但未捕获到异常信息")
    raise last_error


def write_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def process_file(
    csv_path: Path,
    processed: Set[str],
    session: requests.Session,
    args: argparse.Namespace,
    output_path: Path,
    error_path: Path,
    split_dir: Optional[Path],
) -> Tuple[int, int]:
    logging.info("开始处理 %s", csv_path)
    total = 0
    success = 0
    aggregate_resolved = output_path.resolve()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total += 1
            record_hash = row.get("hash")
            if not record_hash:
                logging.warning("第 %s 条缺少 hash，已跳过", total)
                continue
            if record_hash in processed:
                continue
            prompt = build_prompt(row)
            try:
                analysis, raw_text = call_model(
                    session=session,
                    host=args.host,
                    model=args.model,
                    prompt=prompt,
                    max_retries=args.max_retries,
                )
            except Exception as exc:  # noqa: PERF203
                error_payload = {
                    "hash": record_hash,
                    "symbol": row.get("symbol"),
                    "range_label": row.get("range_label"),
                    "source_file": str(csv_path),
                    "error": str(exc),
                    "prompt": prompt,
                }
                if isinstance(exc, ModelResponseError) and exc.raw_response:
                    error_payload["raw_response"] = exc.raw_response
                write_jsonl(error_path, error_payload)
                continue
            output_payload: Dict[str, object] = {
                "hash": record_hash,
                "symbol": row.get("symbol"),
                "security_name": row.get("security_name"),
                "range_label": row.get("range_label"),
                "source": row.get("source"),
                "news_type": row.get("news_type"),
                "published_at": row.get("published_at"),
                "labels": analysis.get("labels"),
                "intensities": analysis.get("intensities"),
                "model": args.model,
                "prompt": prompt,
                "raw_response": raw_text,
            }
            write_jsonl(output_path, output_payload)
            if split_dir is not None:
                split_path = get_split_output_path(split_dir, row.get("range_label"))
                if split_path.resolve() != aggregate_resolved:
                    write_jsonl(split_path, output_payload)
            processed.add(record_hash)
            success += 1
            time.sleep(args.sleep)
    logging.info("文件 %s 处理完成：成功 %s / 总计 %s", csv_path, success, total)
    return total, success


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    split_dir: Optional[Path] = None
    if not args.no_split:
        split_dir = args.split_output_dir
    input_paths = [Path(p) for p in args.input]
    ensure_output(args.output)
    ensure_error_output(args.error_output, args.overwrite)
    processed = load_processed_hashes(args.output, args.overwrite)
    prepare_split_outputs(split_dir, args.overwrite, args.output, args.error_output)
    session = requests.Session()
    total_entries = 0
    total_success = 0
    try:
        for path in input_paths:
            file_total, file_success = process_file(
                csv_path=path,
                processed=processed,
                session=session,
                args=args,
                output_path=args.output,
                error_path=args.error_output,
                split_dir=split_dir,
            )
            total_entries += file_total
            total_success += file_success
    except KeyboardInterrupt:
        logging.warning("用户中断，已保存当前进度（成功 %s 条）", total_success)
    finally:
        session.close()
    logging.info(
        "全部文件处理完毕：累计成功 %s 条 / 遍历 %s 条，结果写入 %s",
        total_success,
        total_entries,
        args.output,
    )


if __name__ == "__main__":
    main()
