# Oracle branch status — 5 November 2026

The root of the `oracle` branch matches the allowlisted deployed Oracle source. `lenovo` preserves the previous GitHub main baseline; it has not been reconciled with the current Lenovo machine. See `/BRANCHES.md` for the separation and commit workflow.

- Deployed: Gemini provider retries/fallback, 8192-token output budget, explicit completion checks, corrected multiline SSE framing, existing hybrid retrieval and reranking safeguard. Backups and observed limits are documented in the dated Oracle snapshot.
- Staged only: free automatic QueryRewriter and prompt/retriever integration; 21 tests passed. OpenAI adapter; 9 simulated provider tests passed, no real API key/credit verified.
- Delivered: seven-scene test CLI and fixture list; saved original-wording retrieval/narrative results. Five scene requests refused for insufficient evidence; two generated narratives still require source review. Corrected-name/book-filter diagnostics also failed for Loken/Jubal and Sharrowkyn/Lucius.
- Diagnosed: useful raw fight passages exist in Horus Rising and Angel Exterminatus. Ranking stitched passages can truncate context beyond the reranker's 512-token input window. The Laer blade example also shows the generic zero cutoff is too strict for some relevant passages. These are unresolved retrieval defects, not evidence the books are missing.

The next proposed repair is to rank raw candidate passages before stitching and validate against all seven requested scenes. That repair has not been implemented or deployed. Provider-class refactoring and automatic GitHub deployment are also pending.
