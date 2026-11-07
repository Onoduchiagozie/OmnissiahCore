"""
Interactive API Client CLI.
Connects to the FastAPI server on port 8000.
"""
import uuid
import json
import os
import sys
import requests

# Fix for "ModuleNotFoundError: No module named 'Core'"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.app_text import app_text

API_BASE_URL = "http://localhost:8000"


def _print_banner():
    print("\n" + "=" * 70)
    print(f"  {app_text['cli']['banner_title']}")
    print("=" * 70)
    print("  /explore    switch to Object Explorer mode (Sync API)")
    print("  /narrate    switch to Narrator mode (Streaming API - Default)")
    print("  /top <n>    change how many chunks are used")
    print("  /window <n> change stitching window size")
    print("  /pool <n>   change candidate pool size")
    print("  /quit       exit")
    print("-" * 70 + "\n")


def _print_sources(chunks: list[dict]):
    if not chunks:
        return
    print("- Sources -")
    for i, c in enumerate(chunks, 1):
        score = c.get("score", 0.0)
        rng = c.get("stitch_range", "")
        line = f"  [{i}] {c.get('source', '?')}  /  {c.get('chapter', '?')}"
        if rng:
            line += f"  ({rng})"
        line += f"  score={score}"
        print(line)
    print()


def _stream_api(payload: dict):
    """Hits the /query/narrate/stream endpoint. ANIMATION REMOVED."""
    url = f"{API_BASE_URL}/query/narrate"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        if response.status_code != 200:
            print(f"\n[ERROR] Server returned status {response.status_code}: {response.text}")
            return []

        sources = []
        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                content = decoded_line[5:]
                if content.startswith(" "):
                    content = content[1:]

                if content == "[DONE]":
                    break
                elif content.startswith("__SOURCES__:"):
                    try:
                        sources = json.loads(content[12:])
                    except json.JSONDecodeError:
                        pass
                else:
                    # INSTANT PRINT - No time.sleep() delay
                    sys.stdout.write(content)
                    sys.stdout.flush()
        print("\n")
        return sources
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")
        return []


def _sync_api(payload: dict):
    """Hits the /query/explore endpoint. ANIMATION REMOVED."""
    url = f"{API_BASE_URL}/query/explore"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        text = data.get("response", "")
        sources = data.get("sources", [])

        # INSTANT PRINT - No for-loop with time.sleep()
        print(text + "\n")

        return sources
    except requests.exceptions.ConnectionError:
        print("\n[ERROR]: Could not connect to API. Is 'python main.py api' running?\n")
        return []


def main():
    mode = "narrator"
    top_k = 10
    candidate_pool = 50
    stitching_window = 3

    # Safely extract arguments and ignore 'cli' so it doesn't trigger a false One-Shot
    args = sys.argv[1:]
    if args and args[0].lower() == "cli":
        args = args[1:]

    # =====================================================================
    # ONE-SHOT CLI OVERRIDE (Only triggers if you pass a real query)
    # =====================================================================
    if len(args) > 0:
        fresh_session = f"cli-shot-{uuid.uuid4().hex[:6]}"
        raw = " ".join(args).strip()

        payload = {
            "query": raw,
            "top_k": top_k,
            "candidate_pool": candidate_pool,
            "stitching_window": stitching_window,
            "session_id": fresh_session
        }

        print("\n" + "=" * 70)
        print("=" * 70)
        print(f"\nQuery: {raw}\n")
        print("Remembrancer (Narrator):\n")

        sources = _stream_api(payload)
        _print_sources(sources)
        sys.exit(0)

    # =====================================================================
    # INTERACTIVE LOOP
    # =====================================================================
    _print_banner()
    loop_session = f"cli-loop-{uuid.uuid4().hex[:6]}"

    while True:
        try:
            # BULLETPROOF PASTE FIX
            print("Petitioner (Paste your query, then press Enter TWICE to submit):")

            lines = []
            while True:
                line = sys.stdin.readline()
                # If they press Enter on an empty line, break the loop and submit
                if not line.strip():
                    break
                lines.append(line.strip())

            raw = " ".join(lines).strip()

        except (EOFError, KeyboardInterrupt):
            print(f"\n{app_text['cli']['sealed_message']}")
            break

        if not raw:
            continue

        if raw.lower() == "/quit":
            print(app_text["cli"]["sealed_message"])
            break

        if raw.lower() == "/explore":
            mode = "explorer"
            print("[Switched to Explorer Mode - Sync API]\n")
            continue

        if raw.lower() == "/narrate":
            mode = "narrator"
            print("[Switched to Narrator Mode - Stream API]\n")
            continue

        if raw.lower().startswith("/top "):
            try:
                top_k = int(raw.split()[1])
                print(f"[top_k set to {top_k}]\n")
            except (IndexError, ValueError):
                pass
            continue

        if raw.lower().startswith("/window "):
            try:
                stitching_window = int(raw.split()[1])
                print(f"[stitching_window set to {stitching_window}]\n")
            except (IndexError, ValueError):
                pass
            continue

        if raw.lower().startswith("/pool "):
            try:
                candidate_pool = int(raw.split()[1])
                print(f"[candidate_pool set to {candidate_pool}]\n")
            except (IndexError, ValueError):
                pass
            continue

        payload = {
            "query": raw,
            "top_k": top_k,
            "candidate_pool": candidate_pool,
            "stitching_window": stitching_window,
            "session_id": loop_session
        }

        print(f"\n{app_text['cli']['consulting_message']}\n")

        if mode == "explorer":
            print("Object Explorer:\n")
            sources = _sync_api(payload)
        else:
            print("Remembrancer (Narrator):\n")
            sources = _stream_api(payload)

        _print_sources(sources)
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()