"""
OmnissiahCore — main.py

Entry point dispatcher. Run from project root.

Usage:
    python main.py cli        → Interactive CLI (query_test.py)
    python main.py api        → FastAPI server on port 8000
    python main.py verify     → Verify index health
    python main.py build      → Build/update FAISS index (Lenovo only)
"""

import sys
import os

COMMANDS = {
    "cli":    ("Scripts/query_test.py",  "Interactive lore query CLI"),
    "api":    ("Api/server.py",           "FastAPI server (uvicorn)"),
    "verify": ("Scripts/verify_db.py",   "Verify index health"),
    "build":  ("Scripts/build_db.py",    "Build FAISS index (Lenovo only)"),
}


def usage():
    print("\nOmnissiahCore — Usage:")
    for cmd, (_, desc) in COMMANDS.items():
        print(f"  python main.py {cmd:<8}  {desc}")
    print()


if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
    usage()
    sys.exit(0)

mode = sys.argv[1]
script, desc = COMMANDS[mode]

print(f"\n[OmnissiahCore] Starting: {desc}\n")

if mode == "api":
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "Api.server:app",
            "--host",
            "localhost",
            "--port",
            "8000",
            "--reload",
        ],
    )
else:
    extra_args = sys.argv[2:]
    os.execv(sys.executable, [sys.executable, script] + extra_args)
