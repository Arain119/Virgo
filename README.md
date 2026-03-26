# Virgo Trader

Virgo Trader 是一个面向研究的量化交易实验项目，核心方法为深度强化学习（PPO）。项目包含 PyQt6 桌面端、训练/调参/回测流水线，以及可选的新闻情绪数据构建能力。

## 免责声明

本项目仅用于研究与教学演示，不构成任何投资建议。使用本项目进行实盘交易或投资决策所产生的任何损失，由使用者自行承担。

## 主要特性

- 桌面端 UI：训练过程监控、图表展示、日志输出，以及内置对话面板（用于查看结果与执行辅助操作）。
- 训练：基于 Stable-Baselines3 的 PPO 训练流程，支持多种策略骨干（`multiscale_cnn` / `transformer` / `cross_attention`）。
- 调参：Optuna 超参搜索，产出 `best_params.json` 与进度文件，便于 UI 侧展示与复用。
- 回测：对指定模型在给定区间进行回测，输出指标与报告（默认写入 `reports/`）。
- 数据与缓存：市场数据与特征工程结果支持本地缓存，重复实验更快。
- 新闻情绪（可选）：爬取新闻并导出结构化数据，支持基于云端 LLM 或本地 Ollama 的情绪标注。

## 系统要求

- Python：>= 3.10（推荐 3.12）。
- UI：Windows 10/11 上体验最好（PyQt6 桌面端）。
- 训练：依赖 PyTorch + stable-baselines3；可使用 CPU 或 CUDA GPU（取决于你的 PyTorch 安装）。

## 日期格式约定

- 训练/回测参数：`YYYYMMDD`（例如 `20240101`）。
- 新闻爬虫参数：`YYYY-MM-DD`（例如 `2024-01-01`，上海时区）。

## 快速开始（Windows，UI）

建议先在独立环境中运行，避免污染全局 Python：

```powershell
# 可选：创建 venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 启动 UI（会按需安装 requirements.txt 中的依赖）
py run.py
```

`run.py` 会在依赖文件 hash 变化时，从 [`requirements.txt`](requirements.txt) 安装 pinned 版本依赖，然后启动桌面端。

## 安装方式（更标准）

如果你希望使用命令行入口或进行开发，推荐在仓库根目录做可编辑安装：

```powershell
pip install -e ".[ui,train,news,dev]"
```

安装后可用的命令行入口（见 `pyproject.toml`）：

| 命令 | 说明 |
| --- | --- |
| `virgo-trader-ui` | 启动桌面 UI |
| `virgo-trader-train` | 训练入口（PPO） |
| `virgo-trader-optimize` | Optuna 调参 |
| `virgo-trader-backtest` | 回测 |
| `virgo-trader-pipeline` | 训练 -> 回测 -> 指标流水线 |
| `virgo-trader-download-data` | 预下载数据（可选构建特征缓存） |
| `virgo-trader-news` | 新闻爬虫 |

## Docker（可选）

如果你更习惯容器化运行（Linux + NVIDIA GPU），可以使用仓库自带的 `Dockerfile` / `docker-compose.yml` 直接跑训练任务（默认把运行时数据挂载到 `./data`）。

```bash
docker compose run --rm virgo-train
```

## 典型工作流（推荐）

1. 预下载数据（可选，但推荐）

```powershell
virgo-trader-download-data --stock_pool sse50 --start_date 20220101 --end_date 20240101 --build_features
```

2. 训练（支持 `--stock_code` 或 `--stock_pool`）

```powershell
virgo-trader-train --stock_pool sse50 --start_date 20220101 --end_date 20240101 --episodes 40 --agent_type cross_attention
```

3. 回测（`--model` 可以是模型名或 `.zip` 路径；日期推荐使用 `YYYYMMDD`）

```powershell
virgo-trader-backtest --model PPO_SSE50_POOL_CROSS_ATTENTION_BASE_20220101_20240101_SEED3 --start_date 20240101 --end_date 20241231
```

4. 一键流水线（训练 -> 回测 -> 指标）

```powershell
virgo-trader-pipeline --stock_pool sse50 --start_date 20220101 --end_date 20240101 --episodes 40 --backtest_start_date 20240101 --backtest_end_date 20241231
```

## 新闻情绪数据（可选）

新闻爬虫支持两种模式：

```powershell
# 增量模式（读取 crawl state，可断点续跑）
virgo-trader-news --mode incremental

# 历史模式（忽略 crawl state；日期格式为 YYYY-MM-DD）
virgo-trader-news --mode historical --start 2024-01-01 --end 2024-12-31
```

情绪标注（高级用法）：

- 云端 DeepSeek（SiliconFlow）：需要环境变量 `SILICONFLOW_API_KEY` 或命令行参数 `--api-key`。
- 本地 Ollama（Fomalhaut）：默认 `http://127.0.0.1:11434`；模型参考：https://www.modelscope.cn/models/Arain119/Fomalhaut

对应脚本位于 `src/virgo_trader/news/label_with_deepseek.py` 与 `src/virgo_trader/news/label_with_fomalhaut.py`，可用 `python -m ... --help` 查看参数。

## 数据与输出目录约定

Virgo Trader 默认不会把运行时产物提交到仓库中（已在 `.gitignore` 忽略）。你可以通过环境变量 `VIRGO_DATA_DIR` 把所有运行产物写到外部目录，保持仓库“干净”。

常见产物包括：

| 路径（相对 `VIRGO_DATA_DIR` 或仓库根目录） | 用途 |
| --- | --- |
| `reports/` | 回测与训练输出、日志、TensorBoard 等 |
| `cache/` | 市场数据与特征工程缓存 |
| `models/user/` | 训练得到的用户模型（可由 UI 删除） |
| `trader_data.db` | 训练会话与回合数据的 SQLite 数据库 |
| `chat_history.json` | UI 对话历史（本地） |
| `news/` | 新闻抓取的运行时数据（按需生成，已忽略） |

PowerShell 示例：

```powershell
# 仅当前终端会话生效
$env:VIRGO_DATA_DIR = "D:\virgo_data"
```

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `VIRGO_DATA_DIR` | 运行时数据根目录（reports/cache/news/models/db 等），不设置则默认写到仓库根目录。 |
| `SILICONFLOW_API_KEY` | 可选：用于云端 LLM 能力（UI 对话与 DeepSeek 情绪标注）。 |
| `VIRGO_SMTP_SERVER` / `VIRGO_SMTP_PORT` / `VIRGO_SENDER_EMAIL` / `VIRGO_SENDER_PASSWORD` | 可选：训练/回测完成邮件通知（建议仅用环境变量配置，不要写入仓库）。 |
| `VIRGO_TEST_RECIPIENT` | 可选：用于邮件通知模块的自检发送。 |

## 预训练模型

- 预训练模型位于 `models/bundled/`，用于评估/回测。
- UI 视 `models/bundled/` 为只读目录，仅允许删除 `models/user/` 下的模型，避免误删预训练模型。
- 预训练模型训练表现、回撤等统计位于 `models/bundled/sse50_pool_best_total_return_2024_metrics.json`。

## 目录结构

```text
<repo-root>/
  src/
    virgo_trader/           核心代码（ui/, training/, backtest/, environment/, utils/, news/）
  models/
    bundled/                仓库自带的预训练模型（UI 中视为只读）
    user/                   训练产生的用户模型输出（运行时生成，默认被 git 忽略）
  tests/                    Pytest 测试
  requirements.txt          `py run.py` 用的 pinned 依赖列表
  pyproject.toml            Python 包/命令行入口定义
  run.py                    桌面启动器
```

## 开发

```powershell
# 代码规范
python -m ruff check .
python -m ruff format .

# 测试
pytest -q
```

## 仓库内容说明

本公开仓库仅包含核心源代码、测试、配置与预训练模型文件。

## 许可证

Apache License 2.0，见 [`LICENSE`](LICENSE)。
