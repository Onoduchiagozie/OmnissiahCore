# Run AFTER reviewing Git state and safely applying the patch.
# Uses the already-configured local Python interpreter; no packages installed.
param(
    [string]$Repo = 'C:\Users\daval\PycharmProjects\OmnissiahCore\OmnissiahCore',
    [string]$Api = 'http://127.0.0.1:8000',
    [switch]$LiveSmoke
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Repo
$python = Join-Path $Repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv\Scripts\python.exe — locate your existing PyCharm interpreter first.' }
$env:OMNISSIAH_ACTIVE_PROFILE = 'lenovo_build'
Write-Host '=== REAL LangGraph dependency (mandatory) ==='
& $python -c 'import langgraph, langgraph.graph; from langgraph.graph import StateGraph; print("Actual LangGraph import OK")'
if ($LASTEXITCODE -ne 0) { throw 'Actual LangGraph is NOT installed. Test doubles are insufficient for acceptance.' }
Write-Host '=== Full offline recovery suite ==='
& $python manage.py test
if ($LASTEXITCODE -ne 0) { throw 'Automated graph tests FAILED.' }
Write-Host '=== Python syntax/bytecode check ==='
& $python -m compileall -q Core\graph_workflow.py Api\routes\graph_routes.py tools\graph_smoke.py tools\graph_eval.py
if ($LASTEXITCODE -ne 0) { throw 'Compilation FAILED.' }
if ($LiveSmoke) {
    Write-Host '=== Local API retrieval-only audit (no model generation) ==='
    & $python tools\graph_smoke.py --api $Api --no-generation
    if ($LASTEXITCODE -ne 0) { throw 'Baseline retrieval smoke FAILED.' }
    & $python tools\graph_eval.py --api $Api --output graph-retrieval-audit.json
    if ($LASTEXITCODE -ne 0) { throw 'Graph retrieval evaluation FAILED.' }
}
Write-Host 'All requested LOCAL checks completed. Live corpus/GPU checks run only with -LiveSmoke.'
