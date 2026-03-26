"""SQLite persistence for training sessions and episode-level logs.

This module is intentionally small and dependency-light so training can record:
- A `training_sessions` row for each training run (model name + parameters).
- An `episodes` row for each episode (portfolio curve + trades).
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .paths import TRADER_DB_PATH, migrate_legacy_files

DB_PATH = str(TRADER_DB_PATH)


def get_db_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    db_path = Path(DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库，创建表"""
    migrate_legacy_files()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 创建训练会话表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_sessions (
            session_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            start_time TEXT NOT NULL,
            train_parameters TEXT
        );
        """)

        # 创建回合数据表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            episode_number INTEGER NOT NULL,
            portfolio_history TEXT,
            trades TEXT,
            FOREIGN KEY (session_id) REFERENCES training_sessions (session_id)
        );
        """)

        conn.commit()
        logging.info("Database initialized successfully.")
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {e}")
    finally:
        if conn:
            conn.close()


def create_training_session(model_name: str, params: dict[str, Any]) -> int | None:
    """在数据库中创建训练会话记录，返回新 session_id。"""
    sql = """ INSERT INTO training_sessions(model_name, start_time, train_parameters)
              VALUES(?,?,?) """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        start_time = params.get("start_time", "")
        params_json = json.dumps(params)
        cursor.execute(sql, (model_name, start_time, params_json))
        conn.commit()
        logging.info(f"Created new training session for model: {model_name}")
        return cursor.lastrowid
    except sqlite3.Error as e:
        logging.error(f"Failed to create training session: {e}")
        return None
    finally:
        if conn:
            conn.close()


def save_episode_data(
    session_id: int,
    episode_number: int,
    portfolio_history: list[float],
    trades_df: pd.DataFrame,
) -> None:
    """保存一个回合的数据到数据库。"""
    sql = """ INSERT INTO episodes(session_id, episode_number, portfolio_history, trades)
              VALUES(?,?,?,?) """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        history_json = json.dumps(portfolio_history)
        # 将DataFrame转换为JSON字符串
        trades_json = trades_df.reset_index().to_json(orient="records")

        cursor.execute(sql, (session_id, episode_number, history_json, trades_json))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(
            f"Failed to save episode data for session {session_id}, episode {episode_number}: {e}"
        )
    finally:
        if conn:
            conn.close()


def get_sessions_for_model(model_name: str) -> list[dict[str, Any]]:
    """根据模型名称获取其所有的训练会话。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_sessions WHERE model_name = ?", (model_name,))
        sessions = cursor.fetchall()
        return [dict(row) for row in sessions]
    except sqlite3.Error as e:
        logging.error(f"Failed to get sessions for model {model_name}: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_episodes_for_session(session_id: int) -> list[dict[str, Any]]:
    """获取一个会话的所有回合数据。"""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM episodes WHERE session_id = ? ORDER BY episode_number", (session_id,)
        )
        episodes = cursor.fetchall()

        results = []
        for row in episodes:
            episode_data = dict(row)
            # 将JSON字符串转换回Python对象
            episode_data["portfolio_history"] = json.loads(episode_data["portfolio_history"])

            trades_df = pd.read_json(episode_data["trades"], orient="records")
            if not trades_df.empty:
                # 确保timestamp列是datetime类型，然后再设置为索引
                trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
                trades_df = trades_df.set_index("timestamp")
            episode_data["trades"] = trades_df

            results.append(episode_data)
        return results
    except sqlite3.Error as e:
        logging.error(f"Failed to get episodes for session {session_id}: {e}")
        return []
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    # 用于测试的简单脚本
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Initializing database...")
    init_db()
    logging.info("Database ready.")
