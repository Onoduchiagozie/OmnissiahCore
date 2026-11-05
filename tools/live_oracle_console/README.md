# Local live Oracle test console

This is a real HTTP client, not a test-results dashboard. It forwards requests through an SSH tunnel to Oracle and displays the unmodified response alongside the answer and source passages.

## Start without a Codex chat

On Dell, double-click `start-oracle-console.cmd` in this directory, then open http://127.0.0.1:18003/ in your normal browser. Keep the command window open while testing. The server starts and restarts its own SSH tunnel. Node.js and OpenSSH are required; the launcher uses the bundled Node runtime if present and otherwise `node` from PATH. Oracle itself does not depend on Node or Codex.

To launch it from PowerShell instead, run `./start-oracle-console.cmd` from this directory. The equivalent manual commands are:

```powershell
$env:ORACLE_TUNNEL_PORT = "18002"
$env:ORACLE_TEST_PORT = "18003"
$env:ORACLE_SSH_AUTOSTART = "1"
node .\server.cjs
```

On the original Dell, if node is not in PATH, the installed executable can also run independently of the Codex app:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" .\server.cjs
```

Open http://127.0.0.1:18003/ in your normal browser. Keep that PowerShell window running; Ctrl+C stops the console and its SSH tunnel. If the Oracle connection drops, the server retries the tunnel and the page refreshes connection status every ten seconds. Do not paste an API key into the page. It uses the provider already configured on Oracle.

## Use

Type your own short question and click Ask Oracle. Scene narration streams readable text. Focus and Explore return JSON answers. Inspect shows actual retrieved passages without an LLM call; expand Read retrieved passage to examine the evidence. Exact request and raw response panels are included.

A refusal or provider error is shown as such. HTTP success does not mean a scene was accurately answered.

## Verified scope

Playwright reached the actual server, submitted the Loken/Jubal question to Inspect, received HTTP 200 and six passage groups, and confirmed the raw response included the inspection. It then submitted a Ferrus Manus/Fulgrim narration: HTTP 200, 7,530 answer characters, six sources, and the stream completion marker after 66.5 seconds. These checks verify the browser route and display, not the factual accuracy of the answer.

The console binds only to loopback and manages the SSH tunnel. It is not a production public frontend or a completed TheForge integration.


