# Session persistence and frontend cache — staged implementation

## Scope and current status

This change is source-only in development branches. Oracle and the hosted TheForge
site have not been updated. Passing unit tests does not establish working
authentication, real retrieval accuracy, or end-to-end persistence.

- Backend: `Core/conversation_store.py`, `Api/services/runtime_service.py`,
  `Api/routes/system_routes.py`.
- Frontend: separate TheForge review branch
  `codex/oracle-frontend-integration-20261106`.
- Oracle production remains the previous in-memory-only runtime.

## What is implemented

1. Every TheForge workspace generates an unpredictable session ID, forwards it
   to FastAPI, and preserves that ID for multiple questions in the same page
   instance. New tabs and page reloads receive new sessions.
2. Missing `session_id` on the staged Python runtime generates a distinct
   anonymous ID instead of sharing the old `default` conversation.
3. Optional SQLite store persists only the most recent four completed
   question/answer pairs for up to seven days since the last update.
   It holds no FAISS vectors, full books, or LangGraph checkpoints.
4. `OMNISSIAH_CONVERSATION_DB` opts in to SQLite; when unset, the backend
   continues using in-process memory.
5. Public `GET /memory` and `DELETE /memory` access is disabled unless a
   private `OMNISSIAH_MEMORY_ADMIN_TOKEN` is configured; then the
   `X-Omnissiah-Memory-Admin` header is required.
6. A single generation slot loads memory after acquiring the semaphore and
   writes the updated history before releasing it, preventing stale snapshots
   from sequential same-session requests.
7. TheForge's 10-minute generated-answer cache keys include session,
   conversation revision, query, book and sources. The legacy cache route
   skips caching if no conversation revision is explicitly supplied.
   Narration streams remain uncached. Explore/Focus UI endpoints are not
   routed through this answer cache.

## Why this is not yet a complete multi-user chat product

- An opaque session ID is **not authentication or authorization**.
  Query endpoints are not yet protected by a validated account/session identity.
- SQLite persistence does not provide a conversation sidebar or saved threads
  across browser visits; a permanent user conversation ID and authenticated
  owner mapping are needed.
- The backend's four-turn LLM context window is not a complete message archive.
- A long-lived public service must enforce account-level ownership, expiry,
  request concurrency, rate limits, quota controls and full diagnostic endpoint
  authorization before enabling chat persistence.
- LangGraph checkpoints are not implemented by this patch.
- TheForge's page-scoped revision cannot detect another client modifying
  the same server-side session. Server-side revision enforcement is needed
  before general cross-device/parallel chat caching.

## Safe local verification

On Dell in a **separate** checkout of OmnissiahCore's development branch:

```powershell
py -3 -m unittest discover -s Tests -p test_conversation_store.py -v
py -3 -m py_compile Core/conversation_store.py Api/services/runtime_service.py Api/routes/system_routes.py
```

To check persistence without starting FastAPI or loading the book corpus:

```powershell
@'
import tempfile
from pathlib import Path
from Core.conversation_store import ConversationStore
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "chat.sqlite3"
    first = ConversationStore(path)
    first.save("forge-local-12345678", [{"query":"Who is Loken?","response":"test answer"}])
    second = ConversationStore(path)
    print(second.get("forge-local-12345678"))
'@ | py -3 -
```

On a separate TheForge review checkout:

```powershell
dotnet restore Tests/TheForge.Tests/TheForge.Tests.csproj
dotnet test Tests/TheForge.Tests/TheForge.Tests.csproj --filter "FullyQualifiedName!~WorkspaceBrowserTests&FullyQualifiedName!~PublishedStylesTests&FullyQualifiedName!~LiveOracleConnectionTests"
```

These are **offline** tests; they do not call Gemini, start Oracle services, or
validate accuracy of passages. GitHub Actions also runs the respective suites.

## Production approval gate

Before enabling backend persistence:
- Back up the exact deployed source/config and verify restore steps.
- Choose a private DB path *outside the Git checkout* and provision appropriate
  service-user directory permissions; never put database contents in Git.
- Configure production session identity/authorization, secret management and
  diagnostic route access without revealing any keys in chat.
- Reconcile the review branch with Dell's local uncommitted frontend changes.
- Validate real browser-to-Oracle sessions and completion behaviour.
- Obtain explicit permission to deploy and restart.

Activation is via private server environment, for example:

`OMNISSIAH_CONVERSATION_DB=/opt/omnissiah/data/private/conversations.sqlite3`

Do **not** set this on production until the access-control gate is complete.
