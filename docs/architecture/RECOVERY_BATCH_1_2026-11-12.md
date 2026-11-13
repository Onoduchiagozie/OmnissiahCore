# Recovered graph integration: batch 1

Date: 12 November 2026. Status: local, uncommitted integration; NOT deployed.

## Provenance

Isolated checkout: recovery-20261112/OmnissiahCore, branch codex/recovery-20261112.
Base main: 668b91a1b08ea0e9932a52fed8e0820fa7fabf26.
Merged development source: origin/dev/retrieval-repair-20261108 (merge not committed).
Resolved whole-file line-ending conflicts in system_routes.py and retriever.py,
retaining main's /version endpoint and static device resolver as well as development changes.

Original recovered packages remain untouched in C:/Users/daval/Downloads/langrapgh.
V2 patch SHA256: 4199881286CA127B72722D82998B50622D956B3659792CB75EF9D255D8D9A907.
V2.1 patch SHA256: 533C03F5CC412FA3E20EA9B5D3CB3E765279678ABE77EFBDE6AC9ADD11AB9D24.
Independent copies are under recovery-20261112/packages.

## Changes beyond the recovered package

- Api/server.py: register graph endpoints only when OMNISSIAH_GRAPH_ENABLED=1.
  Optional dependency stays in requirements-graph.txt, not requirements-runtime.txt.
- Api/routes/graph_routes.py: resolve unique anonymous sessions through RuntimeService,
  return the resolved ID, classify invalid IDs as HTTP 400.
- Hold one LLM slot across memory load, generation, optional verification and accepted-only
  memory save; always release it. Inspection takes no generation slot and saves no turns.
- When the underlying retrieval graph owns recovery, disable the outer EvidenceGraph retry
  loop. Otherwise retain the recovered bounded three-variant loop.
- Core/graph_workflow.py: reconsider combined passages before the context budget,
  so later relevant passages can displace full earlier irrelevant context.
  This uses lexical relevance, NOT semantic proof or a measured accuracy improvement.
- Require real LangGraph in the integration test suite; remove silent fake-runtime fallback.
- Add five regression cases for unique sessions, invalid IDs, memory slot ordering,
  nested retries and explanatory evidence recovery after a full first result.
- Update graph CI to exercise these tests with the backend's FastAPI/Pydantic versions.

## Executed verification

Windows, Python 3.10.11, real langgraph 1.2.20, fastapi 0.136.1,
pydantic 2.13.3, httpx 0.28.1. Optional tracing explicitly disabled for test runs.

- Original unmodified V2.1 package: 46 tests passed with REAL LangGraph
  (initial standalone run used fastapi 0.162.5 / pydantic 2.14.3).
- Integrated graph suite: 51 passed with backend-pinned FastAPI/Pydantic.
- Retrieval graph suite: 7 passed.
- Recovery suite: 5 passed.
- Source-boundary/inspection invariants: 4 passed.
- Conversation store suite: 7 passed.
- Total integrated batch: 74 tests passed.
- pip check: no broken requirements.
- git diff --check: passed; no unresolved merge entries.

These tests use fake corpus/model dependencies. They do NOT establish real retrieval
accuracy, GPU compatibility, production startup, user authentication or frontend integration.

Reproduce from this checkout using an isolated environment:

    python -m pip install -r requirements-graph-test.txt
    python -m unittest discover -s Tests/unit -p "test_graph_*.py"
    python -m unittest discover -s Tests -p "test_langgraph_retrieval.py"
    python -m unittest discover -s Tests -p "test_retrieval_recovery.py"
    python -m unittest discover -s Tests -p "test_retrieval_invariants.py"
    python -m unittest discover -s Tests -p "test_conversation_store.py"

## Remaining integration gates

The generation graph remains a synchronous JSON experiment. The existing SSE endpoints
have not been rewired to it. Deadline/cancellation propagation, provider failures,
actual session ownership, full history and API-key/role/rate-limit design remain work.
The legacy retrieval adapter still applies lexical gates and character truncation.
The separate causal/event-recall and truthful-SSE branches require reconciliation.
Do not treat this batch as the final backend or deploy it to Oracle.
TheForge's feature-refactored UI must receive adapted streaming fixes, not old duplicate pages.
Lenovo unpushed changes and real corpus evidence still need inspection.
