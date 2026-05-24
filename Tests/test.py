#!/usr/bin/env python3
"""
OmnissiahCore Test Suite
9 queries across 3 endpoints (3 per endpoint)
Based on actual indexed files from Warhammer 40K/Horus Heresy corpus
"""

import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://localhost:8000"
SESSION_ID = "test-session-001"

# Test queries based on actual indexed lore
TESTS = {
    "/query/inspect": [
        {
            "name": "Inspect: Fulgrim's Corruption",
            "query": "What corrupted Fulgrim and the Emperor's Children Legion?"
        },
        {
            "name": "Inspect: Istvaan V Dropsite Massacre",
            "query": "What happened at Istvaan V during the Horus Heresy?"
        },
        {
            "name": "Inspect: Iron Hands and Ferrus Manus",
            "query": "Describe the relationship between the Iron Hands and their Primarch Ferrus Manus"
        }
    ],
    "/query/explore": [
        {
            "name": "Explore: The Emperor's Children Legion",
            "query": "What was the Emperor's Children Legion known for?"
        },
        {
            "name": "Explore: The Death of Ferrus Manus",
            "query": "How did Ferrus Manus die and what were the consequences?"
        },
        {
            "name": "Explore: Horus Heresy Timeline",
            "query": "What major events defined the Horus Heresy?"
        }
    ],
    "/query/narrate/stream": [
        {
            "name": "Narrate: Fulgrim vs Ferrus Duel",
            "query": "Narrate the duel between Fulgrim and Ferrus Manus at the height of betrayal"
        },
        {
            "name": "Narrate: Siege of Terra",
            "query": "Tell the story of the final assault on Terra during the Horus Heresy"
        },
        {
            "name": "Narrate: The Fall of the Emperor's Children",
            "query": "Recount how the Emperor's Children fell to Chaos and corruption"
        }
    ]
}


def test_inspect() -> Dict:
    """Test /query/inspect endpoint (retrieval diagnostics, no LLM)"""
    print("\n" + "=" * 80)
    print("TESTING: /query/inspect (Retrieval Diagnostics)")
    print("=" * 80)

    results = []
    for test in TESTS["/query/inspect"]:
        print(f"\n[INSPECT] {test['name']}")
        print(f"Query: {test['query']}")

        try:
            response = requests.post(
                f"{BASE_URL}/query/inspect",
                json={
                    "query": test["query"],
                    "session_id": SESSION_ID,
                    "top_k": 15,
                    "candidate_pool": 80,
                    "stitching_window": 6
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Status: 200 OK")
                print(f"  Chunks retrieved: {data.get('chunks_used', 0)}")
                print(f"  Sources found: {len(data.get('sources', []))}")
                print(f"  Response length: {len(data.get('inspection', {}).get('query_terms', []))} terms")
                results.append({
                    "test": test["name"],
                    "status": "PASS",
                    "chunks": data.get('chunks_used', 0),
                    "sources": len(data.get('sources', []))
                })
            else:
                print(f"✗ Status: {response.status_code}")
                print(f"  Error: {response.text[:200]}")
                results.append({"test": test["name"], "status": "FAIL", "error": response.status_code})

        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            results.append({"test": test["name"], "status": "ERROR", "error": str(e)})

        time.sleep(1)

    return {"endpoint": "/query/inspect", "tests": results}


def test_explore() -> Dict:
    """Test /query/explore endpoint (sync response with analysis)"""
    print("\n" + "=" * 80)
    print("TESTING: /query/explore (Object/Entity Analysis - Sync)")
    print("=" * 80)

    results = []
    for test in TESTS["/query/explore"]:
        print(f"\n[EXPLORE] {test['name']}")
        print(f"Query: {test['query']}")

        try:
            response = requests.post(
                f"{BASE_URL}/query/explore",
                json={
                    "query": test["query"],
                    "session_id": SESSION_ID,
                    "top_k": 15,
                    "candidate_pool": 80,
                    "stitching_window": 6
                },
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                response_text = data.get('response', '')
                print(f"✓ Status: 200 OK")
                print(f"  Response length: {len(response_text)} chars")
                print(f"  Chunks used: {data.get('chunks_used', 0)}")
                print(f"  Sources: {len(data.get('sources', []))}")
                print(f"  Preview: {response_text[:150]}...")
                results.append({
                    "test": test["name"],
                    "status": "PASS",
                    "response_length": len(response_text),
                    "chunks": data.get('chunks_used', 0)
                })
            else:
                print(f"✗ Status: {response.status_code}")
                print(f"  Error: {response.text[:200]}")
                results.append({"test": test["name"], "status": "FAIL", "error": response.status_code})

        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            results.append({"test": test["name"], "status": "ERROR", "error": str(e)})

        time.sleep(1)

    return {"endpoint": "/query/explore", "tests": results}


def test_narrate_stream() -> Dict:
    """Test /query/narrate/stream endpoint (streaming tokens + sources)"""
    print("\n" + "=" * 80)
    print("TESTING: /query/narrate/stream (Cinematic Narrative - SSE Stream)")
    print("=" * 80)

    results = []
    for test in TESTS["/query/narrate/stream"]:
        print(f"\n[NARRATE] {test['name']}")
        print(f"Query: {test['query']}")

        try:
            response = requests.post(
                f"{BASE_URL}/query/narrate/stream",
                json={
                    "query": test["query"],
                    "session_id": SESSION_ID,
                    "top_k": 15,
                    "candidate_pool": 80,
                    "stitching_window": 6
                },
                stream=True,
                timeout=120
            )

            if response.status_code == 200:
                tokens = []
                sources_found = False
                done = False

                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8') if isinstance(line, bytes) else line
                        if line.startswith('data: '):
                            content = line[6:]
                            if content == '[DONE]':
                                done = True
                                break
                            elif content.startswith('__SOURCES__:'):
                                sources_found = True
                            elif not content.startswith('[ERROR]'):
                                tokens.append(content)

                print(f"✓ Status: 200 OK")
                print(f"  Tokens streamed: {len(tokens)}")
                print(f"  Sources received: {sources_found}")
                print(f"  Stream completed: {done}")
                print(f"  Narrative preview: {''.join(tokens[:100])[:150]}...")
                results.append({
                    "test": test["name"],
                    "status": "PASS",
                    "tokens": len(tokens),
                    "sources": sources_found,
                    "completed": done
                })
            else:
                print(f"✗ Status: {response.status_code}")
                results.append({"test": test["name"], "status": "FAIL", "error": response.status_code})

        except Exception as e:
            print(f"✗ Exception: {str(e)}")
            results.append({"test": test["name"], "status": "ERROR", "error": str(e)})

        time.sleep(2)

    return {"endpoint": "/query/narrate/stream", "tests": results}


def main():
    print("\n" + "█" * 80)
    print("█ OmnissiahCore Test Suite")
    print("█ 9 Queries across 3 Endpoints (3 per endpoint)")
    print("█" * 80)

    # Check health first
    print("\nChecking system health...")
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code == 200:
            print("✓ System is ready")
        else:
            print(f"✗ Health check failed: {health.status_code}")
            return
    except Exception as e:
        print(f"✗ Cannot reach server: {str(e)}")
        return

    # Run all tests
    all_results = []
    all_results.append(test_inspect())
    all_results.append(test_explore())
    all_results.append(test_narrate_stream())

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_pass = 0
    total_fail = 0
    total_error = 0

    for endpoint_result in all_results:
        endpoint = endpoint_result["endpoint"]
        tests = endpoint_result["tests"]

        endpoint_pass = sum(1 for t in tests if t["status"] == "PASS")
        endpoint_fail = sum(1 for t in tests if t["status"] == "FAIL")
        endpoint_error = sum(1 for t in tests if t["status"] == "ERROR")

        total_pass += endpoint_pass
        total_fail += endpoint_fail
        total_error += endpoint_error

        print(f"\n{endpoint}")
        print(f"  PASS: {endpoint_pass}/3  FAIL: {endpoint_fail}/3  ERROR: {endpoint_error}/3")

    print(f"\nOVERALL: {total_pass}/9 PASS | {total_fail}/9 FAIL | {total_error}/9 ERROR")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()