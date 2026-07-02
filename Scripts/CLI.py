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
# """
# OmnissiahCore — CLI.py
# ============================
# Drop this file in the project root (same level as main.py).
#
# Run with:
#     python query_cli.py
#
# This file talks directly to OmnissiahAgent and OmnissiahRetriever —
# NO FastAPI server required, NO separate terminal, NO HTTP at all.
# It loads the same models the API uses and streams responses straight
# to your terminal, token by token, exactly like ChatGPT.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER-EDITABLE SETTINGS  ← change these if you want different defaults
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# """
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Core.retriever import OmnissiahRetriever
#
# # Default mode: "narrator" or "explorer"
# DEFAULT_MODE   = "narrator"
#
# # Token delay in seconds. .0 = instant.  0.03 = relaxed typewriter.
# TOKEN_DELAY    = 0.03
#
# # Wrap width for Explorer mode responses (characters per line)
# WRAP_WIDTH     = 88
#
# # Book filter: set a book name here to restrict searches, or "" for all books
# DEFAULT_FILTER = ""
#
# # ── STOP EDITING HERE ─────────────────────────────────────────────────────────
#
# import os
# import sys
# import time
# import textwrap
# import threading
# from Core import retriever
#
# ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, ROOT)
#
# os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# os.environ.setdefault("HF_HUB_OFFLINE",      "1")
#
#
# # ── ANSI colours ──────────────────────────────────────────────────────────────
#
# def _enable_win_ansi():
#     if sys.platform == "win32":
#         try:
#             import ctypes
#             ctypes.windll.kernel32.SetConsoleMode(
#                 ctypes.windll.kernel32.GetStdHandle(-11), 7)
#             return True
#         except Exception:
#             return False
#     return True
#
# USE_COLOR = _enable_win_ansi()
#
# def _c(code, t): return f"\033[{code}m{t}\033[0m" if USE_COLOR else t
# def gold(t):  return _c("38;5;214", t)
# def dim(t):   return _c("2",        t)
# def red(t):   return _c("91",       t)
# def green(t): return _c("92",       t)
# def bold(t):  return _c("1",        t)
# def grey(t):  return _c("90",       t)
#
#
# # ── Banner ────────────────────────────────────────────────────────────────────
#
# BANNER = r"""
#    ___  __  ____  ____  ____  ___  ___  __   __  _   _
#   / _ \|  \/  | \| _ \|_ _|/ __||_ _|/ _\ |  || | | |
#  | (_) | |\/| | . | | || | \__ \ | | | (  | __ | |_| |
#   \___/|_|  |_|_|\_|_|_|___|___/|___|\___|_|  |_|\___/
#        C O R E  -  B L A C K   L I B R A R Y   R A G
# """
#
# def print_banner():
#     print(gold(BANNER))
#     cmds = [
#         ("/narrator",       "Stream cinematic narration  [default]"),
#         ("/explorer",       "Structured object / artifact analysis"),
#         ("/filter <book>",  "Search one book  e.g. /filter Fulgrim"),
#         ("/filter off",     "Search the entire archive"),
#         ("/memory",         "Show last 4 conversation turns"),
#         ("/forget",         "Wipe session memory, start fresh"),
#         ("/mode",           "Show current mode and filter"),
#         ("/help",           "Show this list"),
#         ("/quit",           "Exit"),
#     ]
#     print(bold("  Commands:"))
#     for cmd, desc in cmds:
#         print(f"  {gold(cmd):<30} {dim(desc)}")
#     print()
#
#
# # ── Output helpers ────────────────────────────────────────────────────────────
#
# def typewrite_token(token: str):
#     sys.stdout.write(token)
#     sys.stdout.flush()
#     if TOKEN_DELAY > 0 and token.strip():
#         time.sleep(TOKEN_DELAY)
#
#
# def print_sources(chunks: list):
#     if not chunks:
#         return
#     print(dim("\n  --- Sources " + "-" * 40))
#     for i, c in enumerate(chunks, 1):
#         score = float(
#             c.get("rerank_score") or c.get("query_overlap_score") or
#             c.get("rrf_score")    or c.get("faiss_score") or 0.0
#         )
#         src     = os.path.basename(c.get("source", "?"))
#         chapter = c.get("chapter", "")
#         rng     = c.get("stitch_range", "")
#
#         bar_len = 20
#         filled  = int(min(max(score, 0.0), 1.0) * bar_len)
#         bar     = "#" * filled + "." * (bar_len - filled)
#         bar_col = green(bar) if score > 0 else red(bar)
#
#         line = f"  [{i}] {gold(src)}"
#         if chapter and chapter != "unknown":
#             line += f"  {dim('ch.' + str(chapter))}"
#         if rng:
#             line += f"  {dim(rng)}"
#         line += f"\n      {bar_col} {grey(f'{score:.4f}')}"
#         print(line)
#     print(dim("  " + "-" * 52 + "\n"))
#
#
# def spinner_until(event: threading.Event, message: str):
#     frames = ["|", "/", "-", "\\"]
#     i = 0
#     while not event.is_set():
#         sys.stdout.write(f"\r  {gold(frames[i % 4])} {dim(message)}")
#         sys.stdout.flush()
#         time.sleep(0.12)
#         i += 1
#     sys.stdout.write("\r" + " " * (len(message) + 10) + "\r")
#     sys.stdout.flush()
#
#
# # ── Load the Core (runs once on startup) ─────────────────────────────────────
#
# def load_core():
#     """
#     Initialise OmnissiahRetriever directly — the same thing the API server
#     does on startup. Loads FAISS, metadata.json, embedding model, BM25,
#     and optional CrossEncoder. Takes 10-60s depending on hardware.
#     """
#     done = threading.Event()
#     t = threading.Thread(
#         target=spinner_until,
#         args=(done, "Loading archive  —  FAISS index, embeddings, BM25 ..."),
#         daemon=True
#     )
#     t.start()
#
#     try:
#         retriever = OmnissiahRetriever()
#         done.set()
#         t.join()
#         return retriever
#     except FileNotFoundError as e:
#         done.set(); t.join()
#         print(red(f"\n  [ERROR] Required file not found:\n  {e}"))
#         print(dim("  Run 'python main.py build' to build the index first.\n"))
#         sys.exit(1)
#     except Exception as e:
#         done.set(); t.join()
#         print(red(f"\n  [ERROR] Failed to load archive:\n  {e}\n"))
#         raise
#
#
# def make_agent(retriever, mode: str, session_memory: list):
#     """Create a fresh OmnissiahAgent, injecting existing session memory."""
#     from Core.agent import OmnissiahAgent
#     agent = OmnissiahAgent(retriever=retriever, mode=mode)
#     agent._memory = list(session_memory)
#     return agent
#
#
# # ── Query runners ─────────────────────────────────────────────────────────────
#
# def run_narrator(retriever, query: str, book_filter: str,
#                  session_memory: list):
#     """Stream narrator response token by token. Returns (memory, chunks)."""
#     agent     = make_agent(retriever, "narrator", session_memory)
#     chunks    = []
#     full_text = []
#
#     print()  # blank line before response starts
#
#     try:
#         for token in agent.ask_stream(
#             query,
#             book_filter=book_filter or None,
#             source_filter=None,
#         ):
#             # ask_stream yields string tokens.
#             # The sources come back as a list at the very end.
#             if isinstance(token, list):
#                 chunks = token
#                 continue
#             full_text.append(token)
#             typewrite_token(token)
#
#         print()  # newline after last token
#
#     except KeyboardInterrupt:
#         print(gold("\n\n  [stopped by user]"))
#
#     except Exception as e:
#         print(red(f"\n  [ERROR during generation] {e}\n"))
#
#     # Update session memory
#     updated = list(session_memory) + [{"query": query, "response": "".join(full_text)}]
#     return updated[-4:], chunks  # keep last 4 turns, matches Core/agent.py:237
#
#
# def run_explorer(retriever, query: str, book_filter: str,
#                  session_memory: list):
#     """Synchronous explorer query. Waits then typewriters. Returns (memory, chunks)."""
#     agent = make_agent(retriever, "explorer", session_memory)
#
#     done = threading.Event()
#     t = threading.Thread(
#         target=spinner_until,
#         args=(done, "Consulting the Omnissiah ..."),
#         daemon=True
#     )
#     t.start()
#
#     response = ""
#     chunks   = []
#
#     try:
#         response, chunks = agent.ask(
#             query,
#             book_filter=book_filter or None,
#             source_filter=None,
#         )
#         done.set(); t.join()
#
#         print()
#         wrapped = textwrap.fill(response, width=WRAP_WIDTH, replace_whitespace=False)
#         for char in wrapped:
#             sys.stdout.write(char)
#             sys.stdout.flush()
#             if TOKEN_DELAY > 0 and char.strip():
#                 time.sleep(TOKEN_DELAY)
#         print("\n")
#
#     except KeyboardInterrupt:
#         done.set(); t.join()
#         print(gold("\n\n  [stopped by user]"))
#
#     except Exception as e:
#         done.set(); t.join()
#         print(red(f"\n  [ERROR] {e}\n"))
#
#     updated = list(session_memory) + [{"query": query, "response": response}]
#     return updated[-4:], chunks
#
#
# # ── Memory display ────────────────────────────────────────────────────────────
#
# def print_memory(session_memory: list):
#     if not session_memory:
#         print(dim("\n  No memory yet.\n"))
#         return
#     print(dim(f"\n  --- Memory ({len(session_memory)} turn(s)) " + "-" * 30))
#     for i, turn in enumerate(session_memory, 1):
#         q_preview = turn["query"][:120]
#         r_preview = turn["response"][:200].replace("\n", " ")
#         print(f"\n  {gold(f'[{i}]')} {dim('You:')}  {q_preview}")
#         print(f"       {dim('Core:')} {grey(r_preview)}"
#               f"{'...' if len(turn['response']) > 200 else ''}")
#     print(dim("\n  " + "-" * 52 + "\n"))
#
#
# # ── Main ──────────────────────────────────────────────────────────────────────
#
# def main():
#     import uuid
#
#     print_banner()
#     print(f"  {dim('Initialising...')}")
#     retriever = load_core()
#     print(f"  {green('+')} Archive loaded and ready.\n")
#
#     mode           = DEFAULT_MODE
#     book_filter    = DEFAULT_FILTER
#     session_memory = []
#     session_id     = uuid.uuid4().hex[:10]
#
#     mode_label   = gold("NARRATOR") if mode == "narrator" else gold("EXPLORER")
#     filter_label = gold(book_filter) if book_filter else dim("all books")
#     print(f"  {dim('Session')} {gold(session_id)}  "
#           f"{dim('|')}  {dim('Mode')} {mode_label}  "
#           f"{dim('|')}  {dim('Filter')} {filter_label}")
#     print(f"  {dim('Type')} {gold('/help')} {dim('for commands')}\n")
#
#     while True:
#         try:
#             raw = input(f"{gold('>')} {bold('Petitioner:')} ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print(f"\n\n  {gold('>')} {dim('The Omnissiah watches. Farewell.')}\n")
#             break
#
#         if not raw:
#             continue
#
#         low = raw.lower()
#
#         if low in ("/quit", "/exit", "/q"):
#             print(f"\n  {gold('>')} {dim('The Omnissiah watches. Farewell.')}\n")
#             break
#
#         if low in ("/help", "/?"):
#             print_banner()
#             continue
#
#         if low == "/narrator":
#             mode = "narrator"
#             print(f"\n  {green('+')} Mode -> {gold('NARRATOR')} (streaming)\n")
#             continue
#
#         if low == "/explorer":
#             mode = "explorer"
#             print(f"\n  {green('+')} Mode -> {gold('EXPLORER')} (structured analysis)\n")
#             continue
#
#         if low.startswith("/filter "):
#             val = raw[8:].strip()
#             if val.lower() == "off":
#                 book_filter = ""
#                 print(f"\n  {green('+')} Filter cleared — searching {gold('all books')}\n")
#             else:
#                 book_filter = val
#                 print(f"\n  {green('+')} Filter -> {gold(book_filter)}")
#                 print(f"  {dim('Searching only books matching:')} {gold(book_filter)}\n")
#             continue
#
#         if low == "/memory":
#             print_memory(session_memory)
#             continue
#
#         if low in ("/forget", "/clear"):
#             session_memory = []
#             print(f"\n  {green('+')} Memory cleared.\n")
#             continue
#
#         if low == "/mode":
#             ml = gold("NARRATOR") if mode == "narrator" else gold("EXPLORER")
#             fl = gold(book_filter) if book_filter else dim("all books")
#             print(f"\n  Mode   -> {ml}\n  Filter -> {fl}\n")
#             continue
#
#         # ── Run a query ────────────────────────────────────────────────────────
#         label = "Remembrancer (Narrator)" if mode == "narrator" else "Object Explorer"
#         print(f"\n  {gold('>')} {dim(label)}")
#
#         if mode == "narrator":
#             session_memory, chunks = run_narrator(
#                 retriever, raw, book_filter, session_memory)
#         else:
#             session_memory, chunks = run_explorer(
#                 retriever, raw, book_filter, session_memory)
#
#         print_sources(chunks)
#
#
# if __name__ == "__main__":
#     main()