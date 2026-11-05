@echo off
setlocal
set "ORACLE_TEST_PORT=18003"
set "ORACLE_TUNNEL_PORT=18002"
set "ORACLE_SSH_AUTOSTART=1"
set "NODE_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
if not exist "%NODE_EXE%" set "NODE_EXE=node"
echo Starting the local Oracle test page at http://127.0.0.1:18003/
echo Keep this window open while testing. Press Ctrl+C to stop.
"%NODE_EXE%" "%~dp0server.cjs"
if errorlevel 1 pause

