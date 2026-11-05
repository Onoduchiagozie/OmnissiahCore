# Local live Oracle test console

This is a real HTTP client, not a test-results dashboard. It forwards requests through an SSH tunnel to Oracle and displays the unmodified response alongside the answer and source passages.

## Start without a Codex chat

Use two ordinary Windows PowerShell windows. Node.js is required for the small console server; Oracle itself does not depend on Node or Codex.

Window 1 — keep the SSH tunnel running. Ports 18002/18003 avoid the temporary 18000/18001 ports used in the original chat:

```powershell
ssh -i "$env:USERPROFILE\.ssh\oracle_dell" -N -L 127.0.0.1:18002:127.0.0.1:8000 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 ubuntu@129.151.163.64
```

Window 2 — from this directory:

```powershell
$env:ORACLE_TUNNEL_PORT = "18002"
$env:ORACLE_TEST_PORT = "18003"
node .\server.cjs
```

On the original Dell, if node is not in PATH, the installed executable can also run independently of the Codex app:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" .\server.cjs
```

Open http://127.0.0.1:18003/ in your normal browser. Keep both PowerShell commands running; Ctrl+C stops each. Do not paste an API key into the page. It uses the provider already configured on Oracle.

## Use

Type your own short question and click Ask Oracle. Scene narration streams readable text. Focus and Explore return JSON answers. Inspect shows actual retrieved passages without an LLM call; expand Read retrieved passage to examine the evidence. Exact request and raw response panels are included.

A refusal or provider error is shown as such. HTTP success does not mean a scene was accurately answered.

## Verified scope

Playwright reached the actual server, submitted the Loken/Jubal question to Inspect, received HTTP 200 and six Horus Rising passage groups, and confirmed displayed passage text matched the raw response. No generation call was made for this console check. Generated narration through this new console has not been verified.

The console binds only to loopback and requires the SSH tunnel. It is not a production public frontend or a completed TheForge integration. No standalone PowerShell launcher has been implemented.
