# Oracle production snapshot — 5 November 2026

This folder records the code actually deployed on Oracle. The repository root remains the existing main-branch code. It is not a merged Lenovo/Oracle application.

`app/` contains an allowlisted source snapshot; `manifest.json` records hashes and source encodings. Runtime data, models, databases, credentials, and environment files are excluded.

The deployed changes include Gemini retries and fallback, synchronous/streamed routing, explicit generation-completion checks, multiline SSE framing, an 8192-token Oracle generation allowance, and earlier retrieval reranking safeguards. See `CHANGELOG.md` for observed tests and unresolved defects. Generic reranker cutoff calibration and retrieval recall remain unresolved; previous HTTP success checks did not establish complete or accurate answers.

Future staged changes are recorded separately. No code was written to the Lenovo or deployed to Oracle by this GitHub archival commit. Do not copy the repository root to Oracle as a deployment.
