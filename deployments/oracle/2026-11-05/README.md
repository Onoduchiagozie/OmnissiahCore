# Oracle source snapshot — 5 November 2026

This directory records the deployed Oracle source and its original hashes and encodings. On the named `oracle` branch, these allowlisted files also appear at repository root. On the historical `codex/oracle-stabilization-20261105` review branch, the root was preserved from main.

`app/` excludes runtime data, models, databases, corpus material and secrets. `manifest.json` records source file hashes; the GitHub copy uses UTF-8 and notes the original UTF-16 requirements encoding. `CHANGELOG.md` records the earlier verification and its limits.

Staged query and OpenAI changes are separate from this deployed snapshot. Refer to `/BRANCHES.md` and `/docs/oracle/REVIEW_STATUS_2026-11-05.md` for current status. The `lenovo` branch is only the pre-existing GitHub baseline until the actual Lenovo working tree is compared.
