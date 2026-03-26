"""Centralized path and storage conventions for Virgo Trader.

The application can store runtime artifacts (models, reports, caches) either
under the project root (default) or an external directory via `VIRGO_DATA_DIR`.
This module also provides a best-effort migration for legacy file locations.
"""

import os
from pathlib import Path
from typing import Optional

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _find_project_root(start: Path) -> Path:
    """Best-effort discovery of the repository root.

    In a source checkout (including `src/` layout), the project root contains
    `pyproject.toml` (and typically `.git`). When installed as a wheel, those
    markers may not exist; in that case we fall back to the current working
    directory so the package does not try to write into site-packages.
    """

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root(PACKAGE_DIR)
VIRGO_DATA_DIR = os.environ.get("VIRGO_DATA_DIR", "").strip()
DATA_ROOT = Path(VIRGO_DATA_DIR).expanduser().resolve() if VIRGO_DATA_DIR else PROJECT_ROOT
USING_EXTERNAL_DATA_ROOT = bool(VIRGO_DATA_DIR)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    return _ensure_dir(path)


REPORTS_DIR = DATA_ROOT / "reports" if USING_EXTERNAL_DATA_ROOT else PROJECT_ROOT / "reports"
OPTIMIZATION_DIR = REPORTS_DIR / "optimization"
LOGS_DIR = REPORTS_DIR / "logs"
TENSORBOARD_DIR = REPORTS_DIR / "tensorboard"
# Runtime model outputs are stored outside the package directory to keep the source tree clean.
# We keep user-generated outputs separate from bundled pre-trained checkpoints.
MODELS_DIR = (
    DATA_ROOT / "models" / "user" if USING_EXTERNAL_DATA_ROOT else PROJECT_ROOT / "models" / "user"
)

# Bundled pre-trained models shipped with the repo (read-only from the UI perspective).
BUNDLED_MODELS_DIR = PROJECT_ROOT / "models" / "bundled" / "SSE50_POOL_best_total_return_2024"

# Default storage for app state is under the data root (or project root when `VIRGO_DATA_DIR` is unset).
TRADER_DB_PATH = (
    DATA_ROOT / "trader_data.db" if USING_EXTERNAL_DATA_ROOT else PROJECT_ROOT / "trader_data.db"
)
CHAT_HISTORY_PATH = (
    DATA_ROOT / "chat_history.json"
    if USING_EXTERNAL_DATA_ROOT
    else PROJECT_ROOT / "chat_history.json"
)

# Local caches (market data, feature engineering artifacts, etc.).
CACHE_DIR = DATA_ROOT / "cache" if USING_EXTERNAL_DATA_ROOT else PROJECT_ROOT / "cache"
MARKET_DATA_CACHE_DIR = CACHE_DIR / "market_data"
FEATURE_CACHE_DIR = CACHE_DIR / "features"
SENTIMENT_CACHE_DIR = CACHE_DIR / "sentiment"

BEST_PARAMS_PATH = OPTIMIZATION_DIR / "best_params.json"
PROGRESS_FILE = OPTIMIZATION_DIR / "optimization_progress.json"
STUDY_DB_PATH = OPTIMIZATION_DIR / "ppo-stock-trading-study.db"
RESEARCH_LOG_PATH = LOGS_DIR / "research.log"

LEGACY_BEST_PARAMS_PATH = PROJECT_ROOT / "best_params.json"
LEGACY_PROGRESS_FILE = PROJECT_ROOT / "optimization_progress.json"
LEGACY_STUDY_DB_PATH = PROJECT_ROOT / "ppo-stock-trading-study.db"
LEGACY_RESEARCH_LOG_PATH = PROJECT_ROOT / "research.log"


def _migrate_legacy_file(legacy_path: Path, new_path: Path) -> None:
    if not legacy_path.exists() or new_path.exists():
        return

    new_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        legacy_path.replace(new_path)
    except OSError:
        data = legacy_path.read_bytes()
        new_path.write_bytes(data)
        try:
            legacy_path.unlink()
        except OSError:
            pass


def migrate_legacy_files() -> None:
    _migrate_legacy_file(LEGACY_BEST_PARAMS_PATH, BEST_PARAMS_PATH)
    _migrate_legacy_file(LEGACY_PROGRESS_FILE, PROGRESS_FILE)
    _migrate_legacy_file(LEGACY_STUDY_DB_PATH, STUDY_DB_PATH)
    _migrate_legacy_file(LEGACY_RESEARCH_LOG_PATH, RESEARCH_LOG_PATH)


def find_existing_best_params_path() -> Optional[Path]:
    if BEST_PARAMS_PATH.exists():
        return BEST_PARAMS_PATH
    if LEGACY_BEST_PARAMS_PATH.exists():
        return LEGACY_BEST_PARAMS_PATH
    return None


def find_existing_progress_file() -> Optional[Path]:
    if PROGRESS_FILE.exists():
        return PROGRESS_FILE
    if LEGACY_PROGRESS_FILE.exists():
        return LEGACY_PROGRESS_FILE
    return None


def find_existing_study_db_path() -> Optional[Path]:
    if STUDY_DB_PATH.exists():
        return STUDY_DB_PATH
    if LEGACY_STUDY_DB_PATH.exists():
        return LEGACY_STUDY_DB_PATH
    return None


def find_existing_research_log_path() -> Optional[Path]:
    if RESEARCH_LOG_PATH.exists():
        return RESEARCH_LOG_PATH
    if LEGACY_RESEARCH_LOG_PATH.exists():
        return LEGACY_RESEARCH_LOG_PATH
    return None


def _is_within_dir(path: Path, base_dir: Path) -> bool:
    try:
        path.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return False
    except OSError:
        return False
    return True


def get_model_search_dirs() -> list[Path]:
    """Return directories searched for `.zip` models (in priority order)."""
    candidates = [MODELS_DIR, BUNDLED_MODELS_DIR]
    out: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def list_model_zip_paths() -> list[Path]:
    """List unique model zip files across all search dirs (first match wins).

    Notes:
    - Models may be stored directly under `models/` or grouped into subfolders
      (for example `models/bundled/SSE50_POOL_best_total_return_2024/*.zip`).
      We therefore search recursively.
    """
    seen_names: set[str] = set()
    found: list[Path] = []
    for base_dir in get_model_search_dirs():
        if not base_dir.is_dir():
            continue
        for model_path in sorted(base_dir.rglob("*.zip"), key=lambda p: p.as_posix().lower()):
            if model_path.name in seen_names:
                continue
            seen_names.add(model_path.name)
            found.append(model_path)
    return sorted(found, key=lambda p: p.name.lower())


def resolve_model_zip_path(model_name: str) -> Path:
    """Resolve a model name (with or without `.zip`) to a concrete file path.

    The lookup searches recursively under model search dirs to support grouped
    model folders.
    """
    if not model_name:
        raise ValueError("model_name is required.")
    filename = model_name if model_name.lower().endswith(".zip") else f"{model_name}.zip"
    for base_dir in get_model_search_dirs():
        direct = base_dir / filename
        if direct.exists():
            return direct
        matches = sorted(base_dir.rglob(filename), key=lambda p: p.as_posix().lower())
        if matches:
            return matches[0]
    dirs = ", ".join(str(d) for d in get_model_search_dirs())
    raise FileNotFoundError(f"Model not found: {filename} (searched: {dirs})")


def is_user_model_path(path: Path) -> bool:
    """Whether a model path belongs to MODELS_DIR (safe to delete from the UI)."""
    return _is_within_dir(path, MODELS_DIR)
