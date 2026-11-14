# OmnissiahCore

OmnissiahCore is the backend behind [The Kaban Project](https://legion.runasp.net/), a Warhammer 40,000 lore search and storytelling app. Ask a question about the setting or describe a scene the way you remember it. The app searches an indexed collection of books, finds passages that bear on your question, and uses them to write a response with source references.

## What you can do

- **Explorer** investigates lore questions. Try “Why did Horus turn traitor?” to see what the books reveal about his motives, the events leading up to his betrayal, and how others understood it.
- **Narrator** retells a scene using the passages it finds. Try “Magnus breaks the wards while trying to warn the Emperor,” with *A Thousand Sons* selected as a book filter.
- **Scene Archive** lets you browse saved chronicles and look up a character or scene name. TheForge serves this collection from its own SQLite database. Browsing and searching the archive require no model call.

Sources help you see which books and passages a response draws on. Coverage varies by question and by the collection available to the backend.

## What happens when you ask

1. **Plan the search.** The LangGraph workflow can break a broad question into smaller searches for the people, events, and explanations it needs to cover.
2. **Find and rank passages.** Keyword search (BM25) helps find names and phrases, while vector search (FAISS) looks for related meaning. The system combines the results, reranks the strongest candidates, and can pull in neighbouring passages for context.
3. **Check the evidence.** If an important part of the question is still missing, the workflow can search again for that specific gap. When the available passages only support part of an answer, the response can explain what remains uncertain.
4. **Write and stream the answer.** The FastAPI backend streams the response, stage updates, and source references to [TheForge](https://github.com/Onoduchiagozie/TheForge), the separate ASP.NET Core and Blazor Server frontend. Search activity can show stages such as searching, checking evidence, and retrying when the backend emits them.

## How it runs

The same Python code can run on a local machine with a compatible GPU and an OpenAI-compatible model server such as LM Studio, or on Oracle with a hosted model provider such as Gemini. Local and hosted runs can differ with the selected model, index, and configuration. Keep credentials, machine paths, and private data outside the repository.

`main` is the permanent branch. Use a temporary branch for a change, test it, merge it, then delete the branch. Publishing to Oracle is a separate step.

## First run on a new machine

Install Git, Python 3.10 or 3.12, and OpenSSH. No Codex installation is required.
Use your normal GitHub authentication to clone this private repository:

    git clone https://github.com/Onoduchiagozie/OmnissiahCore.git
    cd OmnissiahCore
    python manage.py init
    python manage.py setup

On Windows use .venv\Scripts\python.exe for subsequent commands.
On Linux/macOS use .venv/bin/python. Activate the environment if you prefer
to type python. If Windows python opens the Store, install Python from python.org
and enable its PATH option; do not use a Codex runtime as your permanent Python.

The initial setup installs the small test dependencies. Unit/provider/release
tests work without a GPU, corpus, model downloads, API credentials or billing:

    python manage.py test
    python manage.py scan

For the real backend, install its runtime dependencies:

    python manage.py setup --full --build

That defaults to CPU PyTorch. For a supported GPU, use the wheel index appropriate
to your GPU and driver from https://pytorch.org/get-started/locally/ :

    python manage.py setup --full --build --torch-index YOUR_OFFICIAL_PYTORCH_INDEX

GPU availability depends on the OS, PyTorch build, driver and device; a machine
having a GPU does not guarantee CUDA support. CPU querying remains available.
FAISS runs on CPU; embeddings/reranking can use the supported GPU.

## Private settings and data

init creates ~/.config/omnissiahcore/settings.json OUTSIDE the source checkout.
Edit that file to select your local model, private paths and machine settings.
The examples in config/ contain no real credentials.

- active_profile: lenovo_build for local build/query; dell_query for Oracle query.
- data_root: parent containing Db/ and Data/raw_pdfs/.
- profiles: partial overrides of the safe defaults in config.json.
- environment: optional string settings such as GEMINI_API_KEY or HF_HOME.
- Existing process environment variables override the private environment file.
- OMNISSIAH_PRIVATE_CONFIG selects a different private JSON file.
- OMNISSIAH_DATA_DIR overrides the data root.
- OMNISSIAH_ACTIVE_PROFILE selects the profile without editing tracked files.

For the local OpenAI-compatible server, load your chosen model in LM Studio (or
another compatible local service) and set its model identifier and chat URL.
The local setting does not require a paid OpenAI API. Oracle uses its existing
Gemini environment credentials; this workflow does not enable billing.

Copy the matching private Db/faiss.index and Db/metadata.json together from your
existing installation, or build a new index from your own documents. Keep their
embedding model/settings compatible. Optional scene database/index/metadata and
book manifests must also be copied to retain those features. Model caches are
downloaded on first use unless offline mode is explicitly enabled.

The source checkout alone does NOT include books, indexes, model weights, API
keys, SSH private keys or private machine notes. Back those up separately.
Removing tracked data now does not erase older Git history.

    python manage.py doctor
    python manage.py api --reload

Open http://127.0.0.1:8000/docs to paste queries directly.
Local startup binds only to loopback and never starts a public ngrok tunnel.
Point a local TheForge instance to http://127.0.0.1:8000/ to test edited backend
code; leaving its backend URL on Oracle tests the deployed backend instead.

    python manage.py smoke
    python manage.py cli

smoke checks health and retrieval without spending LLM quota. Passing it is not
proof that an answer is accurate. Use tools/verify_scenes.py to inspect the
existing real question cases and review returned passages before generation.

## Build a private index

Use the build profile, install --build dependencies, and put documents in the
private Data/raw_pdfs directory:

    python manage.py build

OCR/archive formats additionally need Tesseract, Poppler, 7-Zip and/or Calibre on
PATH. Override unusual locations through OMNISSIAH_TESSERACT,
OMNISSIAH_POPPLER_PATH, OMNISSIAH_7ZIP and OMNISSIAH_CALIBRE in private settings.
ONNX acceleration is optional and requires a compatible export and its optional
packages. Never run build on Oracle's production index.

## Edit, review and merge

Start from a clean, synchronized checkout. Preserve any outstanding local changes
before switching branches.

    git switch main
    git pull --ff-only origin main
    git switch -c codex/my-change

Edit, then run tests, doctor and the live local smoke check as applicable.
Stage specific source files (not git add .), review the staged diff and scan it:

    git add Core/the_file_you_changed.py
    git diff --cached
    python manage.py scan
    git commit -m "Describe the change"
    git push -u origin codex/my-change

On GitHub, open a pull request targeting main. Read the diff and the checks.
The automated Windows/Linux checks cover configuration, provider completion,
release packaging and rollback simulations. They do not run a GPU or the private
full corpus. Test those separately before merging affected code.

Merge the pull request, then delete its branch on GitHub. On your computer:

    git switch main
    git pull --ff-only origin main
    git branch -d codex/my-change
    git fetch --prune

No permanent oracle, lenovo, local or remote branches are needed. Machine settings
select behavior. GitHub's origin is a repository address, not an Oracle server.

## Publish main to Oracle

Copy config/publish.example.json to ~/.config/omnissiahcore/publish.json and fill
in your SSH host, existing private key path, app root and service name. Keep it
outside Git. The server needs Python, the working app virtual environment and
permission for the SSH user to run sudo -n systemctl for this service.

Publishing is an explicit command, not an automatic side effect of pushing code.
This keeps SSH keys on your machine and avoids storing production secrets in CI.

    python manage.py status
    python manage.py publish

publish requires clean main exactly matching origin/main, reruns tests and the
tracked-file scan, and uploads only allowlisted runtime source from that commit.
Preparation compiles the candidate and checks its configuration against existing
data paths. It creates a timestamped, exact source backup ON ORACLE and reports
the files that would change. Production keeps running.

After reviewing the prepared commit, apply it explicitly:

    python manage.py publish --apply FULL_MAIN_COMMIT_SHA

Activation refuses changed live source or active/queued generation requests,
stops the service, installs the source, records release.json, starts the service,
and waits for health plus the matching /version commit. Failed readiness restores
the exact previous source and restarts it. A failure to restart the previous
service is surfaced as an error; inspect server logs rather than assuming success.

    python manage.py status
    python manage.py smoke --url YOUR_PUBLIC_API_URL

To restore the backup path printed by the deployment:

    python manage.py rollback --backup /absolute/backup/path

Rollback changes the running version. status makes this visible; revert the bad
change through a new pull request before the next publish so main and production
converge again. Source deployments do not install packages, rebuild indexes or
migrate data. Dependency/data changes need their own tested preparation.
