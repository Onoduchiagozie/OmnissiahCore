# Recovery validation — 12 November 2026

Run python manage.py test in the configured environment. Dell results: 152 unit/HTTP/graph cases plus 23 retrieval/session cases passed. Real LangGraph 1.2.20 and FastAPI are installed. ML/provider/corpus fixtures are synthetic. No production query or paid model call was made.

Coverage: provider fallback, empty/truncated output, truthful SSE terminals, all four streaming HTTP routes, cancelled waiters, stream cleanup, session separation/retention, filters, raw ranking, bounded recovery, independent style/scope, release packaging and rollback. Evaluation question counts are not counts of proven lore answers.

Remaining:
- Preserve/reconcile Lenovo's unpushed work.
- Start the GPU/corpus runtime with private settings.
- Review Nuceria, Fulgrim/Ferrus, Magnus and unrelated held-out passages; distinguish recall from generation.
- Measure experimental retrieval modes on the complete index.
- Verify Oracle's deployed identity before release.
- Agree/implement account, API-key, role and rate-limit scope before public persistent multi-user features.

No main merge, push, remote branch deletion or deployment has occurred.
