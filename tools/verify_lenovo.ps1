# Read-only preflight for the real Lenovo checkout; reports are written outside the repo.
# NEVER git pull/reset/clean/apply from this script.
param(
    [string]$Repo = 'C:\Users\daval\PycharmProjects\OmnissiahCore\OmnissiahCore',
    [string]$Patch = ''
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Repo)) { throw "Project directory not found: $Repo" }
Set-Location -LiteralPath $Repo
$root = (& git rev-parse --show-toplevel)
if ($LASTEXITCODE -ne 0 -or -not $root) { throw "Not a Git checkout: $Repo" }
$reportDir = Join-Path $env:USERPROFILE ('OmnissiahPreflight_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
"Git root: $root" | Set-Content (Join-Path $reportDir 'SUMMARY.txt')
"Current branch/dirty work:" | Add-Content (Join-Path $reportDir 'SUMMARY.txt')
& git status --short --branch | Tee-Object -FilePath (Join-Path $reportDir 'git-status.txt')
& git branch -avv | Out-File -FilePath (Join-Path $reportDir 'git-branches.txt')
& git log -1 --format='%H %ad %s' --date=iso | Out-File -FilePath (Join-Path $reportDir 'latest-commit.txt')
& git diff --stat | Out-File -FilePath (Join-Path $reportDir 'unstaged-stat.txt')
& git diff --cached --stat | Out-File -FilePath (Join-Path $reportDir 'staged-stat.txt')
& git ls-files Data Mobi Db | Measure-Object -Line | Select-Object -ExpandProperty Lines | Out-File -FilePath (Join-Path $reportDir 'tracked-corpus-count.txt')
$pythonPath = Join-Path $Repo '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $pythonPath) {
    & $pythonPath --version
    & $pythonPath -c 'import importlib.util; print("LangGraph installed:", importlib.util.find_spec("langgraph") is not None)'
} else { Write-Warning 'No .venv\Scripts\python.exe at root; inspect the existing environment before creating one.' }
if ($Patch) {
    if (-not (Test-Path -LiteralPath $Patch)) { throw "Patch not found: $Patch" }
    & git apply --check -- "$Patch"
    if ($LASTEXITCODE -eq 0) { Write-Host 'PATCH CHECK PASSED (patch was NOT applied)' }
    else { Write-Warning 'Patch does not apply cleanly; do not force it.' }
}
Write-Host "Saved read-only Git report at $reportDir"
Write-Host 'No Git content, books, indexes, or Python source were changed.'
