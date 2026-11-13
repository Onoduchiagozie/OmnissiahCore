# Experimental graph integration

The original V2/V2.1 packages have been reconciled into the recovery branch.
Do not apply their patches again. See docs/architecture/REPAIR_BRANCH_RECONCILIATION_2026-11-12.md for provenance.

## Develop and test locally

Python 3.10+: run python manage.py setup, then .venv/Scripts/python.exe manage.py test.
On Linux use .venv/bin/python. Test setup installs real LangGraph and lightweight API dependencies, without ML weights or a corpus. The same manage.py test command runs in the main CI workflow.

For a real local API install runtime dependencies using manage.py setup --full with the correct PyTorch wheel index. Configure data/model/provider settings outside Git using manage.py init-private and manage.py doctor. First inspect Lenovo's existing environment, unpushed code and private data; do not overwrite them.

## Two separate graphs

- RetrievalGraph is selected by retrieval.langgraph_retrieval in private configuration. It interprets, retrieves, ranks raw chunks, stitches, assesses and attempts bounded recovery. Existing generation routes can use it.
- EvidenceGraph adds planning, grading, generation and optional model verification. Set OMNISSIAH_GRAPH_ENABLED=1 before starting the API to register POST /query/graph and /query/graph/inspect. These are JSON routes, not token SSE.
- OMNISSIAH_GRAPH_VERIFY=1 enables an additional provider call. It has not been activated on Oracle. A model verifier is a consistency check, not proof.

Only one retrieval recovery path is selected. EvidenceGraph does not multiply retries when that retriever already expands/retries. Private settings can instead select retrieval_quality for the causal/chronicle experiment; the metadata fallback additionally requires event_anchor_recall. Experiments remain disabled by default pending real-corpus evaluation.

## Coverage and limits

The graph tests require real LangGraph: no substitute runtime. Explicit filters, source-safe stitching, separate sessions, accepted-only memory, bounded retries, raw ranking and truthful stream completion have automated coverage.

Lexical gates and anchor scores do not prove factual correctness. Review the actual Nuceria/Fulgrim/Magnus passages on Lenovo. No chronology database, audio pipeline, account authentication, per-user limits or graph checkpoint store is supplied by this integration.

First run tools/verify_lenovo.ps1 with the actual repository path. It only records state; it does not back up uncommitted source. Preserve and compare that source before applying changes. tools/validate_graph_lenovo.ps1 runs the offline suite; -LiveSmoke additionally requests local retrieval checks.
