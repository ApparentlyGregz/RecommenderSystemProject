"""Launch Révise Malin: the FastAPI backend and the Streamlit interface.

Usage:
    python run.py              API on http://localhost:8000 (docs at /docs), interface on http://localhost:8501
    python run.py --api-only   only the API
    python run.py --lan        listen on every network interface, so another laptop on the same WiFi
                               can open http://<this machine's IP>:8501
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent


def ensure_questions() -> None:
    """Build the practice questions on first launch (the question text is not in Git).

    Downloads the DBE-KT22 files if needed (scripts/download_dbe_kt22.py, about 17 MB),
    then writes data/processed/questions.parquet. Without network the app still runs,
    without the practice mode.
    """
    from src.config import QUESTIONS_FILE

    if QUESTIONS_FILE.exists():
        return
    raw = ROOT / "data" / "raw" / "dbe-kt22"
    try:
        if not (raw / "Questions.csv").exists() or not (raw / "Question_Choices.csv").exists():
            print("First launch: downloading the DBE-KT22 questions for the practice mode...")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "download_dbe_kt22.py")], check=True)
        subprocess.run([sys.executable, "-m", "src.questions"], cwd=ROOT, check=True)
    except (subprocess.CalledProcessError, OSError) as err:
        print(f"Practice mode disabled ({err}). Recommendations still work; "
              "run `python scripts/download_dbe_kt22.py` then `python -m src.questions` to enable it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--lan", action="store_true")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit port")
    args = parser.parse_args()
    os.chdir(ROOT)  # so "api.main" imports and data/feedback.db is found from any working directory
    sys.path.insert(0, str(ROOT))
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    ensure_questions()

    ui = None
    if not args.api_only:
        env = {**os.environ, "REVISE_MALIN_API": f"http://127.0.0.1:{args.port}"}
        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
             "--server.address", host, "--server.port", str(args.ui_port), "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            env=env,
        )
        print(f"Interface: http://localhost:{args.ui_port}")
    try:
        uvicorn.run("api.main:app", host=host, port=args.port)
    finally:
        if ui is not None:
            ui.terminate()
