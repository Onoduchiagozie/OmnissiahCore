"""Oracle Warhammer RAG evaluation: 70% Narrator, 30% Explorer.

100 route tests: cheap, /query/voice-preview, NO retrieval or Gemini calls.
Optional evidence tests: /query/evidence, CPU-heavy, NO Gemini calls.
The results are written to a JSON report for later comparison.

Examples:
    python tools/oracle_benchmark_70_30.py
    python tools/oracle_benchmark_70_30.py --evidence-limit 10
    python tools/oracle_benchmark_70_30.py --evidence-limit 20 --out /tmp/omni-eval.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Tests/data/warhammer_eval_70_30.json"

BOILERPLATE = re.compile(
    r"more (?:horus heresy|warhammer) (?:books|stories) from blacklibrary"
    r"|\bblacklibrary\.com\b"
    r"|\b(?:ISBN|copyright all rights reserved)\b",
    re.IGNORECASE,
)


def request_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def choose_evidence_sample(questions, count):
    """Balanced sample, approximately 70/30, preserving historical regressions."""
    if count <= 0:
        return []
    narrators = [q for q in questions if q["expected_style"] == "narrator"]
    explorers = [q for q in questions if q["expected_style"] == "explorer"]
    narr_count = min(len(narrators), round(count * .7))
    other_count = min(len(explorers), count - narr_count)
    selected = narrators[:narr_count] + explorers[:other_count]
    # Expand deterministically if one group is exhausted.
    for q in questions:
        if len(selected) >= count:
            break
        if q not in selected:
            selected.append(q)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--evidence-limit", type=int, default=0,
                        help="How many actual cross-encoder retrievals to run (default: 0)")
    parser.add_argument("--out", default="/tmp/oracle-eval-70-30.json")
    args = parser.parse_args()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    questions = data["questions"]
    assert len(questions) == 100
    expected = Counter(q["expected_style"] for q in questions)
    assert expected == {"narrator": 70, "explorer": 30}

    base = args.base_url.rstrip("/")
    rows = []
    failures = []
    print("=== 100 ROUTING TESTS: NO GEMINI, NO RETRIEVAL ===", flush=True)
    for i, q in enumerate(questions, 1):
        start = time.monotonic()
        try:
            reply = request_json(base + "/query/voice-preview", {
                "query": q["query"], "retrieval_scope": q["retrieval_scope"]
            }, timeout=30)
            ok = (reply.get("response_style") == q["expected_style"]
                  and reply.get("retrieval_scope") == q["retrieval_scope"]
                  and reply.get("llm_used") is False)
            row = {
                "id": q["id"], "kind": "routing", "passed": ok,
                "expected_style": q["expected_style"],
                "observed_style": reply.get("response_style"),
                "scope": q["retrieval_scope"],
                "elapsed_seconds": round(time.monotonic() - start, 3),
            }
        except Exception as exc:
            row = {"id": q["id"], "kind": "routing", "passed": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
        if i % 10 == 0:
            print(f"{i}/100 routing tested; failures: {len(failures)}", flush=True)

    sample = choose_evidence_sample(questions, min(100, max(0, args.evidence_limit)))
    print(f"\n=== {len(sample)} FINAL-EVIDENCE TESTS: NO GEMINI ===", flush=True)
    for i, q in enumerate(sample, 1):
        start = time.monotonic()
        try:
            reply = request_json(base + "/query/evidence", {
                "query": q["query"],
                "retrieval_scope": q["retrieval_scope"],
                "top_k": 8, "candidate_pool": 60, "stitching_window": 1
            })
            hits = reply.get("hits", [])
            sources = [h.get("source", "") for h in hits]
            hint = q.get("expected_source_hint")
            has_hint = not hint or any(hint.lower() in s.lower() for s in sources)
            has_passages = bool(hits)
            boilerplate_count = sum(
                1 for h in hits if BOILERPLATE.search(h.get("text", ""))
            )
            ok = has_hint and has_passages and boilerplate_count == 0
            row = {
                "id": q["id"], "kind": "evidence", "passed": ok,
                "style": q["expected_style"], "scope": q["retrieval_scope"],
                "primary_source_hint": hint, "hint_found": has_hint,
                "passages": len(hits), "boilerplate_hits": boilerplate_count,
                "sources": sources,
                "preview": [re.sub(r"\s+", " ", h.get("text", ""))[:180] for h in hits],
                "elapsed_seconds": round(time.monotonic() - start, 3),
            }
        except Exception as exc:
            row = {"id": q["id"], "kind": "evidence", "passed": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)
        if not row["passed"]:
            failures.append(row)
        print(f"{i}/{len(sample)} {q['id']} {'PASS' if row['passed'] else 'FAIL'} "
              f"{row.get('sources', row.get('error', ''))}", flush=True)

    report = {
        "dataset": "warhammer_eval_70_30.json",
        "expected_mix": dict(expected),
        "routing_count": 100,
        "evidence_count": len(sample),
        "routing_passed": sum(x["passed"] for x in rows if x["kind"] == "routing"),
        "evidence_passed": sum(x["passed"] for x in rows if x["kind"] == "evidence"),
        "failures": failures,
        "results": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved: {args.out}")
    print(f"Routing: {report['routing_passed']}/100; "
          f"evidence: {report['evidence_passed']}/{len(sample)}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
