#!/usr/bin/env python3
"""
Comprehensive Test Suite for OmnissiahCore RAG
Tests: Health → Sources → Inspect → Query (Narrative) → Stream (Narrative)

Run: python Tests/test_comprehensive.py
Each test runs independently. If one fails, you can run the next.
"""

import json
import sys
import time
from typing import Optional

import httpx

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 3600  # 1 hour for heavy LLM

# ANSI colors for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    """Print a section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")


def print_subheader(text: str):
    """Print a subsection header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}>>> {text}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def print_response(text: str, max_length: int = 500):
    """Print response with truncation"""
    if len(text) > max_length:
        print(f"{Colors.YELLOW}{text[:max_length]}...{Colors.END}")
        print(f"{Colors.YELLOW}[Response truncated. Full length: {len(text)} chars]{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{text}{Colors.END}")


# ============================================================================
# TEST 1: Health Check
# ============================================================================

def test_health() -> bool:
    """Test GET /health endpoint"""
    print_header("TEST 1: Health Check")
    print_info("Endpoint: GET /health")
    print_info("Purpose: Verify system is online and Ollama is accessible")

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{BASE_URL}/health")

        print_info(f"HTTP Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print_success("System is online")
            print(f"  Status: {data.get('status')}")
            print(f"  Profile: {data.get('active_profile')}")
            print(f"  Model: {data.get('ollama_model')}")
            print(f"  Metadata Loaded: {data.get('metadata_loaded')} chunks")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_response(response.text)
            return False

    except httpx.ConnectError:
        print_error("Cannot connect to FastAPI server")
        print_info(f"Make sure the server is running at {BASE_URL}")
        print_info("Run: python main.py api")
        return False
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False


# ============================================================================
# TEST 2: List Sources
# ============================================================================

def test_sources() -> bool:
    """Test GET /sources endpoint"""
    print_header("TEST 2: List Available Sources")
    print_info("Endpoint: GET /sources")
    print_info("Purpose: See what books are indexed")

    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{BASE_URL}/sources")

        if response.status_code == 200:
            data = response.json()
            total = data.get("total", 0)
            sources = data.get("sources", [])

            print_success(f"Found {total} sources")

            # Find Ferrus/Fulgrim related sources
            fulgrim_sources = [s for s in sources if "fulgrim" in s.lower()]
            ferrus_sources = [s for s in sources if "ferrus" in s.lower()]
            horus_sources = [s for s in sources if "horus" in s.lower()]

            if fulgrim_sources:
                print_info("Fulgrim-related sources:")
                for s in fulgrim_sources[:5]:
                    print(f"    • {s}")
                if len(fulgrim_sources) > 5:
                    print(f"    ... and {len(fulgrim_sources) - 5} more")

            if ferrus_sources:
                print_info("Ferrus-related sources:")
                for s in ferrus_sources[:5]:
                    print(f"    • {s}")
                if len(ferrus_sources) > 5:
                    print(f"    ... and {len(ferrus_sources) - 5} more")

            if horus_sources:
                print_info("Horus-related sources:")
                for s in horus_sources[:5]:
                    print(f"    • {s}")

            print_success("Sources retrieved successfully")
            return True
        else:
            print_error(f"Unexpected status: {response.status_code}")
            return False

    except Exception as e:
        print_error(f"Failed to get sources: {e}")
        return False


# ============================================================================
# TEST 3: Inspect Retrieval (No LLM)
# ============================================================================

def test_inspect() -> bool:
    """Test POST /query/inspect endpoint"""
    print_header("TEST 3: Inspect Retrieval")
    print_info("Endpoint: POST /query/inspect")
    print_info("Purpose: See what chunks are retrieved WITHOUT generating (fast)")
    print_info("Query: 'Narrate the full confrontation between Ferrus Manus and Fulgrim'")

    query = {
        "query": "Narrate the full confrontation between Ferrus Manus and Fulgrim from their discussion on Ferrus' ship through the duel to his death",
        "top_k": 15,
        "candidate_pool": 80,
        "stitching_window": 6,
    }

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(f"{BASE_URL}/query/inspect", json=query)

        if response.status_code == 200:
            data = response.json()

            # Extract inspection data
            inspection = data.get("inspection", {})
            faiss_hits = inspection.get("faiss_hits", [])
            bm25_hits = inspection.get("bm25_hits", [])
            stitched_hits = inspection.get("stitched_hits", [])

            print_success("Retrieval inspection completed")
            print_info(f"FAISS hits: {len(faiss_hits)}")
            print_info(f"BM25 hits: {len(bm25_hits)}")
            print_info(f"Stitched hits: {len(stitched_hits)}")

            if stitched_hits:
                print_success("Chunks were retrieved successfully")
                print_info("Top sources for this query:")
                for i, hit in enumerate(stitched_hits[:5], 1):
                    score = hit.get("faiss_score", hit.get("rrf_score", 0.0))
                    source = hit.get("source", "unknown")
                    stitch = hit.get("stitch_range", "")
                    print(f"  [{i}] {source} ({stitch}) - score: {score:.4f}")

                if len(stitched_hits) > 5:
                    print(f"  ... and {len(stitched_hits) - 5} more chunks")

                print_info("Prompt preview:")
                prompt_preview = data.get("prompt_preview", {})
                system = prompt_preview.get("system_prompt", "")
                user = prompt_preview.get("user_message", "")

                print(f"  System prompt length: {len(system)} chars")
                print(f"  User message length: {len(user)} chars")

                print_success("Ready to send to LLM for generation")
                return True
            else:
                print_error("No chunks were retrieved")
                print_info("This might mean:")
                print_info("  1. Query terms don't match any chunks")
                print_info("  2. FAISS vector alignment is poor")
                print_info("  3. Chunks are being filtered out")
                return False

        else:
            print_error(f"Inspect failed with status {response.status_code}")
            print_response(response.text)
            return False

    except httpx.TimeoutException:
        print_error("Inspect timed out (30 seconds)")
        print_info("This shouldn't happen - inspect doesn't call LLM")
        return False
    except Exception as e:
        print_error(f"Inspect failed: {e}")
        return False


# ============================================================================
# TEST 4: Query Sync (Narrator Mode)
# ============================================================================

def test_query_narrator() -> bool:
    """Test POST /query with narrator mode"""
    print_header("TEST 4: Full Query (Narrator Mode - Synchronous)")
    print_info("Endpoint: POST /query")
    print_info("Mode: narrator")
    print_info("Purpose: Get full narrative of Ferrus vs Fulgrim")
    print_info("Warning: This will take 2-5 minutes (30B model is slow)")

    query = {
        "query": "Narrate the full confrontation between Ferrus Manus and Fulgrim from their discussion on Ferrus' ship through the duel to his death",
        "mode": "narrator",
        "top_k": 15,
        "candidate_pool": 80,
        "stitching_window": 6,
        "session_id": "ferrus-fulgrim-test",
    }

    try:
        print_info("Sending query to server...")
        print_info("⏳ Waiting for response (this is SLOW with 30B model)...")

        with httpx.Client(timeout=TIMEOUT) as client:
            start_time = time.time()
            response = client.post(f"{BASE_URL}/query", json=query)
            elapsed = time.time() - start_time

        print_info(f"Response received in {elapsed:.1f} seconds")

        if response.status_code == 200:
            data = response.json()

            query_text = data.get("query", "")
            response_text = data.get("response", "")
            sources = data.get("sources", [])
            chunks_used = data.get("chunks_used", 0)

            print_success("Query completed successfully")
            print_info(f"Chunks used: {chunks_used}")
            print_info(f"Sources cited: {len(sources)}")

            # Print the response
            print_subheader("GENERATED NARRATIVE")
            print_response(response_text, max_length=2000)

            # Print sources
            if sources:
                print_subheader("SOURCES USED")
                for i, src in enumerate(sources[:10], 1):
                    source_name = src.get("source", "unknown")
                    chapter = src.get("chapter", "?")
                    score = src.get("score", 0.0)
                    stitch = src.get("stitch_range", "")
                    print(f"  [{i}] {source_name}")
                    print(f"       Chapter: {chapter}")
                    print(f"       Score: {score:.4f}")
                    print(f"       Range: {stitch}")

                if len(sources) > 10:
                    print(f"  ... and {len(sources) - 10} more sources")

            print_success("Query test completed")
            return True

        elif response.status_code == 400:
            print_error("Bad request (400)")
            print_response(response.text)
            return False
        elif response.status_code == 504:
            print_error("Gateway timeout (504)")
            print_info("Your LLM is too slow or not responding")
            print_info("Check: Is Ollama running? Is the model loaded?")
            return False
        else:
            print_error(f"Unexpected status: {response.status_code}")
            print_response(response.text)
            return False

    except httpx.TimeoutException:
        print_error(f"Request timed out after {TIMEOUT} seconds")
        print_info("Your 30B model might need more time")
        print_info("Try increasing TIMEOUT at the top of this file")
        return False
    except httpx.ConnectError:
        print_error("Cannot connect to server")
        print_info("Make sure FastAPI is running: python main.py api")
        return False
    except Exception as e:
        print_error(f"Query failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# TEST 5: Query Stream (Narrator Mode - Real-time Tokens)
# ============================================================================

def test_query_stream_narrator() -> bool:
    """Test POST /query/stream with narrator mode"""
    print_header("TEST 5: Streaming Query (Narrator Mode - Real-time Tokens)")
    print_info("Endpoint: POST /query/stream")
    print_info("Mode: narrator")
    print_info("Purpose: See narrative being generated token-by-token (like watching film)")
    print_info("Warning: This will take 2-5 minutes (30B model is slow)")

    query = {
        "query": "Narrate the full confrontation between Ferrus Manus and Fulgrim from their discussion on Ferrus' ship through the duel to his death",
        "top_k": 15,
        "candidate_pool": 80,
        "stitching_window": 6,
        "session_id": "ferrus-fulgrim-stream-test",
    }

    try:
        print_info("Connecting to streaming endpoint...")

        with httpx.Client(timeout=TIMEOUT) as client:
            with client.stream("POST", f"{BASE_URL}/query/stream", json=query) as response:
                if response.status_code == 200:
                    print_success("Stream connected, receiving tokens...")
                    print_subheader("STREAMING NARRATIVE (Real-time)")

                    accumulated = ""
                    sources = []
                    token_count = 0

                    # Read and display tokens as they arrive
                    for line in response.iter_lines():
                        if not line:
                            continue

                        if line.startswith("data: "):
                            token_text = line[6:]  # Remove "data: " prefix

                            # Check for sources marker
                            if token_text.startswith("__SOURCES__:"):
                                sources_json = token_text[12:]  # Remove marker
                                try:
                                    sources = json.loads(sources_json)
                                except json.JSONDecodeError:
                                    pass

                            # Check for done marker
                            elif token_text == "[DONE]":
                                print_success("\n✓ Stream complete")
                                break

                            # Check for error
                            elif token_text.startswith("[ERROR]"):
                                print_error(f"\n✗ Stream error: {token_text}")
                                return False

                            # Regular token
                            else:
                                accumulated += token_text
                                token_count += 1
                                # Print without newline to see real-time streaming
                                print(token_text, end="", flush=True)

                    print()  # New line after tokens

                    # Print final stats
                    print_success(f"Streamed {token_count} tokens")
                    print_info(f"Total response length: {len(accumulated)} chars")

                    # Print sources from stream
                    if sources:
                        print_subheader("SOURCES FROM STREAM")
                        for i, src in enumerate(sources[:10], 1):
                            source_name = src.get("source", "unknown")
                            score = src.get("score", 0.0)
                            print(f"  [{i}] {source_name} (score: {score:.4f})")

                    print_success("Stream test completed")
                    return True
                else:
                    print_error(f"Stream failed with status {response.status_code}")
                    return False

    except httpx.TimeoutException:
        print_error(f"Stream timed out after {TIMEOUT} seconds")
        return False
    except httpx.ConnectError:
        print_error("Cannot connect to stream endpoint")
        return False
    except Exception as e:
        print_error(f"Stream failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Run all tests sequentially"""
    print_header("OMNISSIAHCORE COMPREHENSIVE TEST SUITE")
    print_info("Testing Ferrus Manus vs Fulgrim Narrative")
    print_info(f"Server: {BASE_URL}")
    print_info("Each test runs independently - if one fails, you can continue with the next\n")

    tests = [
        ("Health Check", test_health),
        ("List Sources", test_sources),
        ("Inspect Retrieval", test_inspect),
        ("Query (Narrator Mode)", test_query_narrator),
        ("Stream (Narrator Mode)", test_query_stream_narrator),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result

            if result:
                print_success(f"{test_name} PASSED\n")
            else:
                print_error(f"{test_name} FAILED\n")
                # Ask user if they want to continue
                user_input = input(f"{Colors.YELLOW}Continue to next test? (y/n): {Colors.END}")
                if user_input.lower() != "y":
                    print_info("Stopping tests")
                    break

        except KeyboardInterrupt:
            print_info("\nTests interrupted by user")
            break
        except Exception as e:
            print_error(f"Unexpected error in {test_name}: {e}")
            results[test_name] = False

    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.END} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print_success("All tests passed! Your RAG is working.")
        return 0
    else:
        print_error(f"{total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
