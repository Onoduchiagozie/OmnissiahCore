# Oracle review branch — 5 November 2026

This branch records Oracle work without reconciling or modifying Lenovo. The repository root remains the existing main code.

- `deployments/oracle/2026-11-05/app/`: allowlisted deployed Oracle source snapshot and manifest. It includes the actual provider/retrieval/output-completion changes.
- `staging/natural-queries/`: proposed zero-API-cost QueryRewriter, retrieval and prompt integrations. Twenty-one tests passed on Oracle. Not deployed. Generic reranker cutoff and scene retrieval remain unresolved.
- `staging/openai-migration/`: tested Responses adapter and proposed agent integration. Nine mocked tests passed. No key/credit or live model access established; not deployed.
- `tools/verify_scenes.py` and `tools/scene_cases.json`: repeatable seven-scene checks. Default is raw human wording and retrieval-only inspection. Generation requires choosing one case explicitly.
- `docs/oracle/SCENE_VERIFICATION_2026-11-05.md`: endpoints, commands, observed failures and next acceptance criteria.
- `docs/oracle/scene-narrative-results.json`: result metadata only; generated scenes and source text remain on Oracle.

The broad natural-language narrative trial produced two unverified substantive responses and five insufficient-evidence responses. HTTP/SSE success is not evidence of a correct answer. Do not mark the application production-complete or deploy repository-root code over Oracle or Lenovo.

Future provider-class refactoring and GitHub deployment automation are not implemented. Further production edits require user approval and backups. Commits on this branch are review records, not automatic deployments.
