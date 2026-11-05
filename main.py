"""
OmnissiahCoreOld — main.py

Entry point dispatcher. Run from project root.

Usage:
    python main.py cli        → Interactive CLI (CLI.py)
    python main.py api        → FastAPI server on port 8000
    python main.py verify     → Verify index health
    python main.py build      → Build/update FAISS index (Lenovo only)
"""

import sys
import os
import time
import json
import subprocess
import urllib.request

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

NGROK_DOMAIN = "emerald-nativity-guacamole.ngrok-free.dev"
NGROK_PORT = 8000

COMMANDS = {
    "cli":    ("Scripts/CLI.py",  "Interactive lore query CLI"),
    "api":    ("Api/server.py",           "FastAPI server (uvicorn)"),
    "verify": ("Scripts/verify_db.py",   "Verify index health"),
    "build":  ("Scripts/build_db.py",    "Build FAISS index (Lenovo only)"),
}


def usage():
    print("\nOmnissiahCoreOld — Usage:")
    for cmd, (_, desc) in COMMANDS.items():
        print(f"  python main.py {cmd:<8}  {desc}")
    print()


def ngrok_already_running():
    """True if ngrok's local API shows a live tunnel on our domain."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as resp:
            tunnels = json.loads(resp.read()).get("tunnels", [])
            return any(NGROK_DOMAIN in t.get("public_url", "") for t in tunnels)
    except Exception:
        return False


def start_ngrok():
    if ngrok_already_running():
        print(f"[ngrok] already running on {NGROK_DOMAIN}, skipping launch")
        return

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngrok.log")
    log_file = open(log_path, "w", encoding="utf-8")

    popen_kwargs = {"stdout": log_file, "stderr": subprocess.STDOUT}
    if os.name == "nt":
        # New process group so Ctrl+C on the server doesn't also kill ngrok
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        subprocess.Popen(
            ["ngrok", "http", f"--domain={NGROK_DOMAIN}", str(NGROK_PORT)],
            **popen_kwargs,
        )
    except FileNotFoundError:
        print("[ngrok] executable not found on PATH — skipping auto-start")
        log_file.close()
        return

    log_file.close()

    for _ in range(10):
        time.sleep(1)
        if ngrok_already_running():
            print(f"[ngrok] tunnel live: https://{NGROK_DOMAIN}")
            return
    print("[ngrok] did not confirm within 10s — check ngrok.log")


if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    usage()
    sys.exit(0)

mode = sys.argv[1]
script, desc = COMMANDS[mode]

print(f"\n[OmnissiahCore] Starting: {desc}\n")

if mode == "api":
    start_ngrok()
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "Api.server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ],
    )
else:
    extra_args = sys.argv[2:]
    os.execv(sys.executable, [sys.executable, script] + extra_args)