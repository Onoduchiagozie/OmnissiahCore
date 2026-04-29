"""
Editable application and domain strings loader.
"""

import json
import os
import sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_TEXT_PATH = os.path.join(BASE_DIR, "app_text.json")


def _fatal(message: str):
    print(f"[FATAL] {message}")
    sys.exit(1)


def _load_app_text() -> dict:
    if not os.path.exists(APP_TEXT_PATH):
        _fatal(f"app_text.json not found at: {APP_TEXT_PATH}")
    try:
        with open(APP_TEXT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        _fatal(f"app_text.json is malformed: {e}")


app_text = _load_app_text()
