# Oracle and Lenovo branches

## oracle

The root of this branch now contains the allowlisted source actually deployed on Oracle on 5 November 2026. It includes the deployed provider, retrieval and output-completion repairs. `deployments/oracle/2026-11-05/` preserves the audited snapshot and original source hashes. No `.env`, model, corpus, database, or runtime-index assets were added by this work.

`staging/natural-queries/` contains the proposed free QueryRewriter and prompt/retriever changes (21 tests passed). Those staged files are not deployed. The known generic reranker threshold and scene-selection defects remain. `staging/openai-migration/` is also inactive pending a usable API key/credit and real access tests.

Use `tools/verify_scenes.py --case list` to list the seven user-requested cases. Its default mode is retrieval inspection using the original wording; generating a scene requires choosing one case. See `docs/oracle/SCENE_VERIFICATION_2026-11-05.md` for observations and endpoint commands. Five initial live scene requests refused evidence; two generated text but have not been verified accurate.

Oracle itself is still not a Git checkout. Updating this branch does not deploy it. The live systemd service selects `OMNISSIAH_ACTIVE_PROFILE=dell_query`; runtime data and environment configuration stay on Oracle.

## lenovo

Created from the pre-existing GitHub main commit c59aee2973cfb1642b4b209d5380c25d63bcf63c. This is a preserved GitHub baseline, NOT a verified copy of the current Lenovo working tree. The Lenovo machine has not been accessed or modified. Compare and preserve its local changes before using this branch for synchronization.

## Commit workflow

The user requests commits and pushes for authorized code changes. Put Oracle work on `oracle`; preserve `lenovo` independently. Keep deployed versus staged work explicit in commit messages and notes. Do not merge machine profiles blindly, overwrite Lenovo, force-push, commit secrets/corpus/runtime assets, or treat a GitHub push as approval to restart/deploy production.
