#!/usr/bin/env bash
# Oracle deployment: immutable GitHub code, backup, tests, and automatic rollback.
set -Eeuo pipefail
APP="/opt/omnissiah/app"
BASE="3e7c0aba623ae613d98263c8b7f0e73bd9ef7191"
REV="69cb984a0362901ec6cb283627ed3a331a77d4ab"
RAW="https://raw.githubusercontent.com/Onoduchiagozie/OmnissiahCore"
FILES=(
  Core/retriever.py
  Core/retrieval_quality.py
  Api/routes/query_routes.py
  app_text.json
  Tests/unit/test_oracle_chronicle_quality.py
  tools/oracle_chronicle_smoke.py
)
EXISTING=(
  Core/retriever.py
  Core/retrieval_quality.py
  Api/routes/query_routes.py
  app_text.json
)
cd "$APP"
PY="$APP/.venv/bin/python"
test -x "$PY"
STAGE=$(mktemp -d)
BACKUP=""
APPLIED=0

cleanup() { rm -rf "$STAGE"; }
rollback() {
  rc=$?
  trap - ERR
  if [ "$APPLIED" -eq 1 ]; then
    echo "=== Restoring prior production files ==="
    for f in ${FILES[@]}; do
      if [ -f "$BACKUP/$f" ]; then
        mkdir -p "$(dirname "$APP/$f")"
        cp -p "$BACKUP/$f" "$APP/$f"
      else
        rm -f "$APP/$f"
      fi
    done
    sudo systemctl restart omnissiah-api.service || true
  fi
  echo "FAILED (exit $rc). Backup: $BACKUP"
  exit "$rc"
}
trap cleanup EXIT
trap rollback ERR

echo "=== 1. Download immutable GitHub revision ==="
for f in ${FILES[@]}; do
  mkdir -p "$STAGE/new/$(dirname "$f")"
  curl --retry 2 -fsSL "$RAW/$REV/$f" -o "$STAGE/new/$f"
done
for f in ${EXISTING[@]}; do
  mkdir -p "$STAGE/base/$(dirname "$f")"
  curl --retry 2 -fsSL "$RAW/$BASE/$f" -o "$STAGE/base/$f"
done

echo "=== 2. Verify existing deployment is unchanged ==="
"$PY" - "$APP" "$STAGE/base" "${EXISTING[@]}" <<'PYCODE'
from pathlib import Path
import sys
live, baseline, *files = sys.argv[1:]
mismatches = []
for name in files:
    a = (Path(live) / name).read_bytes().replace(b"\r\n", b"\n")
    b = (Path(baseline) / name).read_bytes().replace(b"\r\n", b"\n")
    if a != b:
        mismatches.append(name)
if mismatches:
    sys.exit("ABORT: Production code differs from expected baseline: " + ", ".join(mismatches))
print("Production baseline verified.")
PYCODE

echo "=== 3. Validate syntax and run unit tests before changing production ==="
"$PY" -m py_compile \
  "$STAGE/new/Core/retriever.py" \
  "$STAGE/new/Core/retrieval_quality.py" \
  "$STAGE/new/Api/routes/query_routes.py" \
  "$STAGE/new/tools/oracle_chronicle_smoke.py"
"$PY" -m json.tool "$STAGE/new/app_text.json" >/dev/null
cp Tests/unit/test_oracle_retrieval_quality.py \
  "$STAGE/new/Tests/unit/test_oracle_retrieval_quality.py"
(
  cd "$STAGE/new"
  PYTHONPATH="$STAGE/new" "$PY" -m unittest discover \
    -s Tests/unit -p 'test_oracle_*quality.py' -v
)

echo "=== 4. Create backup and install changes ==="
mkdir -p /opt/omnissiah/backups
BACKUP=$(mktemp -d /opt/omnissiah/backups/chronicle-XXXXXXXX)
for f in ${FILES[@]}; do
  if [ -f "$APP/$f" ]; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -p "$APP/$f" "$BACKUP/$f"
  fi
done
APPLIED=1
for f in ${FILES[@]}; do
  install -D -m 644 "$STAGE/new/$f" "$APP/$f"
done

echo "=== 5. Restart API and verify health ==="
sudo systemctl restart omnissiah-api.service
READY=0
for i in $(seq 1 80); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done
test "$READY" -eq 1
curl -fsS http://127.0.0.1:8000/health
echo

echo "=== 6. Verify actual final evidence without spending Gemini tokens ==="
"$PY" tools/oracle_chronicle_smoke.py
echo "=== ORACLE CHRONICLE PATCH VERIFIED ==="
echo "Rollback backup: $BACKUP"
