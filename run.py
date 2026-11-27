"""Launch Révise Malin: the FastAPI backend and the Streamlit interface.

Usage:
    python run.py                 then open http://localhost:8501   (API docs: http://localhost:8000/docs)
    python run.py --lan           also reachable from another machine on the same network

If the DBE-KT22 data has not been prepared yet and its zip is in data/raw/dbe/,
it is prepared first (a few seconds). Ctrl+C stops both servers.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW, EVENTS_FILE, QUESTIONS_FILE, SERVING_FILE, TOPICS_FILE  # noqa: E402


def ensure_data() -> None:
    """Prepare DBE-KT22 if needed, or explain how to get it."""
    if SERVING_FILE.exists() and all(p.exists() for p in (EVENTS_FILE, TOPICS_FILE, QUESTIONS_FILE)):
        return
    if not SERVING_FILE.exists():
        sys.exit(f"Model bundle missing: {SERVING_FILE}. It is in the repository; run `git pull` or retrain (README).")
    raw = DATA_RAW / "dbe" / "2_DBE_KT22_datafiles_100102_csv.zip"
    if not raw.exists():
        sys.exit(f"DBE-KT22 data not found. Download '2_DBE_KT22_datafiles_100102_csv.zip' (README, step 3) into {raw.parent}")
    print("Preparing DBE-KT22 (first launch only)...")
    subprocess.run([sys.executable, "-m", "src.prepare_dbe", "--raw", str(raw)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


def wait_for(url: str, seconds: float = 40) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def lan_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lan", action="store_true", help="listen on all interfaces (demo from another machine)")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=8501)
    args = parser.parse_args()

    ensure_data()
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    api_url = f"http://127.0.0.1:{args.api_port}"
    env = {**os.environ, "REVISE_MALIN_API": api_url, "PYTHONUNBUFFERED": "1"}
    procs = [subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--host", host,
                               "--port", str(args.api_port), "--log-level", "warning"], cwd=ROOT, env=env)]
    try:
        if not wait_for(f"{api_url}/health"):
            sys.exit("The API did not start; see the error above.")
        procs.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
                                       "--server.port", str(args.ui_port), "--server.address", host,
                                       "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                                      cwd=ROOT, env=env))
        wait_for(f"http://127.0.0.1:{args.ui_port}", 30)
        shown = lan_ip() if args.lan else "localhost"
        print(f"\nRévise Malin is running:\n  interface  http://{shown}:{args.ui_port}\n"
              f"  API docs   http://{shown}:{args.api_port}/docs\nPress Ctrl+C to stop.\n")
        while all(p.poll() is None for p in procs):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
