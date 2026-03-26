# Virgo Trader

Virgo Trader 是一个面向研究的量化交易项目，核心方法为深度强化学习（PPO）。项目包含：PyQt6 桌面 UI、训练/调参/回测能力，以及新闻情绪数据构建流水线（可选）。

本仓库已按“软著打包 + 开源发布”的要求做过清理：默认不携带运行时产物（缓存、日志、数据库、回测输出等）。这些数据会写入 `VIRGO_DATA_DIR` 指定目录；如果未设置该变量，则默认写到仓库根目录下（相关目录/文件已在 `.gitignore` 中忽略）。

## 快速开始（Windows）

```powershell
py run.py
```

`run.py` 会按需（仅当依赖文件 hash 变化时）从 `requirements.txt` 安装依赖，然后启动桌面应用。

## 安装（pip）

更标准的 Python 工作流（在仓库根目录进行可编辑安装）：

```powershell
pip install -e ".[ui,train,news,dev]"
```

命令行入口（见 `pyproject.toml`）：

```powershell
virgo-trader-ui
virgo-trader-train --help
virgo-trader-optimize --help
virgo-trader-news --help
virgo-trader-backtest --help
virgo-trader-pipeline --help
virgo-trader-download-data --help
```

## 目录结构

```text
<repo-root>/
  src/
    virgo_trader/           核心代码（ui/, training/, backtest/, environment/, utils/, news/）
  requirements.txt          `py run.py` 用的依赖列表
  models/
    bundled/                仓库自带的预训练模型（UI 中视为只读）
    user/                   训练产生的用户模型输出（可由 UI 删除）
  paper/                    论文相关材料（manuscripts/tables/figures/data）
  tests/                    Pytest 测试
  run.py                    桌面启动器
```

说明：新闻爬虫的运行时产物默认写入 `<repo-root>/news/`（或 `<VIRGO_DATA_DIR>/news/`），该目录按需生成且已被 git 忽略，因此不会污染源代码包结构。

## 模型

- 预训练模型位于 `models/bundled/`，用于评估/回测。
- 预训练模型的训练表现与回撤等统计：`models/bundled/sse50_pool_best_total_return_2024_metrics.json`。
- 训练输出位于 `models/user/`。
- UI 仅允许删除 `models/user/` 下的模型，避免误删预训练模型。

## 新闻情绪数据（可选）

`virgo_trader.news` 可用于抓取新闻并导出可用于情绪建模/训练的结构化数据。

```powershell
# 增量模式（读取 crawl state，可断点续跑）
virgo-trader-news --mode incremental

# 历史模式（忽略 crawl state）
virgo-trader-news --mode historical --start 2020-01-01
```

## 环境变量

- `VIRGO_DATA_DIR`：
  可选的外部数据根目录，用于放置运行时产物（reports、cache、news 数据、user models 等）。未设置时默认使用仓库根目录。

- 邮件通知（不要把凭据写进代码/仓库）：
  - `VIRGO_SMTP_SERVER`, `VIRGO_SMTP_PORT`, `VIRGO_SENDER_EMAIL`, `VIRGO_SENDER_PASSWORD`
  - 可选：`VIRGO_TEST_RECIPIENT`

## 许可证

Apache License 2.0，见 `LICENSE`。
