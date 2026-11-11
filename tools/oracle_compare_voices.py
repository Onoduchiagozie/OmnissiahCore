"""Human-vs-cinematic QA of the same question and evidence scope.

EXPLICIT Gemini usage (2 live LLM calls):
  python tools/oracle_compare_voices.py --generate

Default (no --generate) performs only a no-cost /query/voice-preview check.
Results: /tmp/oracle-voice-comparison.json

This is a manual-quality test: textual style is displayed for review, not
automatically claimed as factual accuracy or AI voice quality.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
QUESTION = "Why did Horus turn traitor?"


def post(endpoint, payload, timeout=360):
    request = urllib.request.Request(
        BASE + endpoint, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true",
                        help="Make two real Gemini calls to compare generated prose.")
    parser.add_argument("--query", default=QUESTION)
    parser.add_argument("--scope", choices=("focus", "wide"), default="wide")
    parser.add_argument("--out", default="/tmp/oracle-voice-comparison.json")
    args = parser.parse_args()
    results = {
        "question": args.query,
        "scope": args.scope,
        "llm_calls": 0,
        "styles": {},
    }
    for style in ("explorer", "narrator"):
        payload = {
            "query": args.query,
            "response_style": style,
            "retrieval_scope": args.scope,
            "session_id": "voice-compare-" + style,
            "top_k": 8,
            "candidate_pool": 40,
            "stitching_window": 1,
        }
        preview = post("/query/voice-preview", payload, timeout=30)
        if preview.get("response_style") != style or preview.get("retrieval_scope") != args.scope:
            raise RuntimeError(f"Wrong style/scope routing: {preview}")
        print(f"\n=== {style.upper()} / {args.scope} ===", flush=True)
        if not args.generate:
            print("Routing only: PASS. No Gemini call.")
            results["styles"][style] = {"routing": "pass", "llm_used": False}
            continue

        # The same question and retrieval parameters are supplied to both
        # voices. The retrieved evidence could still differ if indexing is
        # nondeterministic; inspect returned source lists before comparison.
        result = post("/query/respond", payload)
        results["llm_calls"] += 1
        text = result.get("response", "")
        sources = result.get("sources", [])
        results["styles"][style] = {
            "routing": "pass",
            "llm_used": True,
            "response": text,
            "sources": sources,
            "chars": len(text),
        }
        print("Sources:", [x.get("source") for x in sources])
        print(text[:5500] if text else "(EMPTY RESPONSE)")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {args.out}")
    if args.generate:
        values = list(results["styles"].values())
        a = [x.get("source") for x in values[0].get("sources", [])]
        b = [x.get("source") for x in values[1].get("sources", [])]
        print("Same ordered source list:", a == b)
        print("Human review needed for voice, chronology, grounding and completeness.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
