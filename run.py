"""Bootstrap entrypoint for Virgo.

This script optionally installs dependencies from `requirements.txt`
(if the hash changes) and then launches the main Qt application.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TSINGHUA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
TSINGHUA_HOST = "pypi.tuna.tsinghua.edu.cn"
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
STATE_FILE = PROJECT_ROOT / "config" / ".deps_state.json"


def _read_state():
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_state(data):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_requirements():
    if not REQUIREMENTS_FILE.exists():
        return None
    return hashlib.sha256(REQUIREMENTS_FILE.read_bytes()).hexdigest()


def install_dependencies_if_needed():
    req_hash = _hash_requirements()
    if req_hash is None:
        return

    state = _read_state()
    if state.get("requirements_hash") == req_hash:
        print("Dependencies are already installed; skipping.\n")
        return

    print("=" * 50)
    print(" Virgo Trader")
    print("=" * 50)
    print()
    print("[1/1] Installing pinned dependencies (requirements.txt)...")

    pip_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "-r",
        str(REQUIREMENTS_FILE),
        "-i",
        TSINGHUA_INDEX,
        "--trusted-host",
        TSINGHUA_HOST,
    ]

    # Command is constructed by the program (not user input).
    result = subprocess.run(pip_cmd)  # noqa: S603
    if result.returncode != 0:
        print()
        print("Error: dependency installation failed. See the output above.")
        print("Common causes:")
        print("1. Multiple Python environments are installed on the system.")
        print("2. The current `python` command is not the target environment (try `where python`).")
        sys.exit(result.returncode)

    _write_state(
        {
            "requirements_hash": req_hash,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pip_command": " ".join(pip_cmd),
        }
    )
    print("Dependencies installed.\n")


def launch_application():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Suppress noisy DEBUG logs from HTTP clients
    for noisy_logger in [
        "urllib3",
        "urllib3.connectionpool",
        "requests",
        "requests.packages.urllib3",
    ]:
        logger = logging.getLogger(noisy_logger)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    src_dir = PROJECT_ROOT / "src"
    sys.path.insert(0, str(src_dir))

    from virgo_trader import main

    main.main()


def main():
    install_dependencies_if_needed()
    launch_application()


if __name__ == "__main__":
    main()
