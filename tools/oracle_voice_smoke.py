"""Live Oracle API smoke test for automatic style/scope without Gemini quota.

Run: .venv/bin/python tools/oracle_voice_smoke.py
"""
import json
import sys
import urllib.request


CASES = [
    ({"query": "Why did Horus turn traitor?"}, "explorer", "wide"),
    ({"query": "Tell the complete story of Horus falling to Chaos."}, "narrator", "wide"),
    ({"query": "Why did Horus turn traitor?", "retrieval_scope": "focus"}, "explorer", "focus"),
    ({"query": "Narrate the battle at Isstvan V.", "retrieval_scope": "focus"}, "narrator", "focus"),
    ({"query": "What happened on Davin?", "response_style": "narrator"}, "narrator", "wide"),
]


def main():
    failed = False
    for payload, expected_style, expected_scope in CASES:
        request = urllib.request.Request(
            "http://127.0.0.1:8000/query/voice-preview",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.load(response)
        ok = (result.get("response_style") == expected_style and
              result.get("retrieval_scope") == expected_scope and
              result.get("llm_used") is False)
        status = "PASS" if ok else "FAIL"
        print(f"{status} | {result.get('response_style')}/{result.get('retrieval_scope')}"
              f" | {payload['query']}")
        if not ok:
            failed = True
    if failed:
        print("FAIL: One or more voice/scope routing cases.")
        return 1
    print("PASS: All style/scope cases. Generation quality requires separate review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
