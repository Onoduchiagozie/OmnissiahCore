"""Real Oracle non-Horus smoke, max ONE LLM generation.

Three /query/inspect searches cost zero provider tokens.
One /query/narrate request costs provider tokens if Gemini is configured.
This is an integration smoke test, NOT proof of 100-answer lore accuracy.
Run on a GitHub-hosted runner to reach a publicly exposed Oracle tunnel.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

RETRIEVAL_QUERIES = [
    "How did the Ultramarines defend Calth against the Word Bearers?",
    "Explain the influence of the Laer blade on Fulgrim.",
    "Why is Commissar Yarrick important to Ghazghkull?",
]
NARRATIVE_QUERY = (
    "Narrate the duel between Leman Russ and Magnus on Prospero "
    "as a chronological battle scene."
)


def request(base, route, payload, timeout=180):
    url = base.rstrip("/") + route
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "ngrok-skip-browser-warning": "true",
            "User-Agent": "OmnissiahIntegrationSmoke/1.0",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def inspect(base):
    for question in RETRIEVAL_QUERIES:
        print("\nRETRIEVAL:", question, flush=True)
        with request(base, "/query/inspect", {
            "query": question, "top_k": 7, "candidate_pool": 25,
            "stitching_window": 1,
        }, timeout=180) as response:
            data = json.load(response)
        report = data.get("inspection", {})
        hits = report.get("stitched_hits", [])
        sources = list(dict.fromkeys(hit.get("source", "?") for hit in hits))
        print("Top inspect sources:", sources[:7], flush=True)
        if not hits:
            raise AssertionError("Inspect returned no passages")
        print("Inspect returned", len(hits), "pre-rerank stitched passages.", flush=True)


def generate(base):
    print("\nGENERATE:", NARRATIVE_QUERY, flush=True)
    tokens, sources, stages, errors = [], [], [], []
    done = False
    deadline = time.monotonic() + 480
    with request(base, "/query/narrate", {
        "query": NARRATIVE_QUERY,
        "session_id": "nonhorus-live-smoke",
        "top_k": 7, "candidate_pool": 25,
        "stitching_window": 1,
    }, timeout=180) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise AssertionError("The API did not return SSE")
        for raw in response:
            if time.monotonic() >= deadline:
                raise TimeoutError("Generation exceeded eight minutes")
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            payload = line[5:].removeprefix(" ")
            if payload == "[DONE]":
                done = True
                break
            if payload.startswith("[ERROR]"):
                errors.append(payload)
                break
            if payload.startswith("__STAGE__:"):
                stages.append(payload.split(":", 1)[1])
                continue
            if payload.startswith("__SOURCES__:"):
                sources = json.loads(payload.split(":", 1)[1])
                continue
            tokens.append(payload)

    content = "".join(tokens)
    print("Generation chars:", len(content), flush=True)
    print("Stages:", stages, flush=True)
    print("Source count:", len(sources), flush=True)
    print("Source names:", [x.get("source", "?") for x in sources][:8], flush=True)
    print("Sample answer first 180 chars:", " ".join(content[:180].split()), flush=True)
    print("Terminal [DONE] received:", done, flush=True)
    if errors:
        raise AssertionError("; ".join(errors))
    if "failed" in stages or "error" in stages:
        raise AssertionError("Oracle returned a failed stage")
    if not done or "complete" not in stages:
        raise AssertionError("Oracle did not signal successful completion")
    if len(content.strip()) < 100:
        raise AssertionError("Oracle generated fewer than 100 answer characters")
    if not sources:
        raise AssertionError("Oracle generated text without source references")
    print("PASS: one live non-Horus response had text, sources and completion.", flush=True)
    print("Factual relevance and full chronology still require independent review.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    try:
        inspect(args.base)
        if not args.skip_generation:
            generate(args.base)
        return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR {e.code} for {e.url}: {e.read(500)!r}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
