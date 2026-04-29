"""
Interactive CLI for testing the full pipeline locally.
"""

import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.agent import OmnissiahAgent
from Core.app_text import app_text
from Core.prompt import format_debug


def _print_banner():
    print("\n" + "=" * 70)
    print(f"  {app_text['cli']['banner_title']}")
    print("=" * 70)
    print("  /clear      reset session memory")
    print("  /book <x>   filter to book containing substring x")
    print("  /books      list all indexed books")
    print("  /explore    switch to Object Explorer mode")
    print("  /remember   switch back to Remembrancer mode (default)")
    print("  /verbose    toggle debug output")
    print("  /top <n>    change how many chunks are used (default: from config)")
    print("  /window <n> change stitching window size")
    print("  /battle <x> full-context battle/event reconstruction")
    print("  /quit       exit")
    print("-" * 70 + "\n")


def _list_books():
    try:
        r = requests.get("http://localhost:8000/sources", timeout=3)
        sources = r.json()["sources"]
    except Exception:
        from Core.config_loader import paths

        try:
            with open(paths["metadata"], "r", encoding="utf-8", errors="replace") as f:
                meta = json.load(f)
            sources = sorted(set(m.get("source", "?") for m in meta))
        except Exception as e:
            print(f"  Could not load sources: {e}")
            return

    print(f"\n  {len(sources)} indexed sources:\n")
    for s in sources:
        print(f"    {s}")
    print()


def main():
    _print_banner()
    agent = OmnissiahAgent(verbose=False)

    book_filter = None
    verbose = False
    top_k = None
    stitching_window = None

    while True:
        try:
            raw = input("Petitioner: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{app_text['cli']['sealed_message']}")
            break

        if not raw:
            continue

        if raw.lower() == "/quit":
            print(app_text["cli"]["sealed_message"])
            break

        if raw.lower() == "/clear":
            agent.clear_memory()
            print(f"{app_text['cli']['memory_cleared']}\n")
            continue

        if raw.lower() == "/verbose":
            verbose = not verbose
            agent.verbose = verbose
            print(f"[Verbose: {'ON' if verbose else 'OFF'}]\n")
            continue

        if raw.lower() == "/explore":
            agent.mode = "explorer"
            print(f"{app_text['cli']['explore_enabled']}\n")
            continue

        if raw.lower() in ("/remember", "/remembrancer"):
            agent.mode = "remembrancer"
            print(f"{app_text['cli']['remember_enabled']}\n")
            continue

        if raw.lower() == "/books":
            _list_books()
            continue

        if raw.lower().startswith("/book"):
            parts = raw.split(maxsplit=1)
            if len(parts) > 1:
                book_filter = parts[1]
                print(f"[Book filter: '{book_filter}']\n")
            else:
                book_filter = None
                print("[Book filter cleared - searching all books.]\n")
            continue

        if raw.lower().startswith("/top "):
            try:
                top_k = int(raw.split()[1])
                print(f"[top_k set to {top_k}]\n")
            except (IndexError, ValueError):
                print("[Usage: /top 8]\n")
            continue

        if raw.lower().startswith("/window "):
            try:
                stitching_window = int(raw.split()[1])
                print(f"[stitching_window set to {stitching_window}]\n")
            except (IndexError, ValueError):
                print("[Usage: /window 3]\n")
            continue

        if raw.lower().startswith("/battle "):
            topic = raw[len("/battle "):].strip()
            print(f"\n[{app_text['cli']['battle_mode_label']} - {topic}]")
            print(f"{app_text['cli']['battle_consulting_message']}\n")
            response, chunks = agent.ask(
                query=topic,
                book_filter=book_filter,
                top_k=12,
                candidate_pool=40,
                stitching_window=3,
            )
            print(f"Remembrancer:\n{response}\n")
            if verbose and chunks:
                print(format_debug(topic, chunks, response))
            else:
                _print_sources(chunks)
            continue

        print(f"\n{app_text['cli']['consulting_message']}\n")
        response, chunks = agent.ask(
            query=raw,
            book_filter=book_filter,
            top_k=top_k,
            stitching_window=stitching_window,
        )

        mode_label = "Object Explorer" if agent.mode == "explorer" else "Remembrancer"
        print(f"{mode_label}:\n{response}\n")

        if verbose:
            print(format_debug(raw, chunks, response))
        else:
            _print_sources(chunks)


def _print_sources(chunks: list[dict]):
    if not chunks:
        return
    print("- Sources -")
    for i, c in enumerate(chunks, 1):
        score = round(
            c.get("rerank_score")
            or c.get("query_overlap_score")
            or c.get("rrf_score")
            or c.get("faiss_score")
            or 0.0,
            4,
        )
        rng = c.get("stitch_range", "")
        line = f"  [{i}] {c.get('source', '?')}  /  {c.get('chapter', '?')}"
        if rng:
            line += f"  ({rng})"
        line += f"  score={score}"
        print(line)
    print()


if __name__ == "__main__":
    main()
