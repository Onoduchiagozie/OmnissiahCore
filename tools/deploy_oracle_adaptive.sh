#!/usr/bin/env bash
# Oracle ARM64: guarded rollout of chronicle fixes + adaptive voice.
# Requires deployed PR #2 OR PR #3 baseline; refuses drift.
set -Eeuo pipefail
APP=/opt/omnissiah/app
RAW=https://raw.githubusercontent.com/Onoduchiagozie/OmnissiahCore
REV=b3c542db401da82cd1bf23aa3b9844acd179ede2
BASES=(3e7c0aba623ae613d98263c8b7f0e73bd9ef7191 6130df296f83c0bb04c79335bb3709f8830fd40c)
FILES=(
  Core/retriever.py Core/retrieval_quality.py Core/agent.py
  Core/prompt.py Core/response_style.py
  Api/models.py Api/routes/query_routes.py Api/services/runtime_service.py
  app_text.json
  Tests/unit/test_oracle_chronicle_quality.py
  Tests/unit/test_oracle_adaptive_voice.py
  tools/oracle_chronicle_smoke.py
  tools/oracle_voice_smoke.py
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
    echo "=== ROLLBACK: restoring original files ==="
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
  echo "DEPLOYMENT FAILED (code $rc). Backup: $BACKUP"
  exit "$rc"
}
trap cleanup EXIT
trap rollback ERR

echo "=== Downloading pinned source ==="
for f in ${FILES[@]}; do
  mkdir -p "$STAGE/new/$(dirname "$f")"
  curl -fsSL --retry 2 "$RAW/$REV/$f" -o "$STAGE/new/$f"
done
for idx in 0 1; do
  base="${BASES[$idx]}"
  for f in ${FILES[@]}; do
    mkdir -p "$STAGE/base$idx/$(dirname "$f")"
    code=$(curl -LsS --retry 2 --output "$STAGE/base$idx/$f" --write-out '%{http_code}' "$RAW/$base/$f")
    if [ "$code" = 404 ]; then
      rm -f "$STAGE/base$idx/$f"
    elif [ "$code" != 200 ]; then
      echo "Cannot verify baseline $base/$f: HTTP $code"
      exit 1
    fi
  done
done

echo "=== Verifying live code matches PR #2 or PR #3 (no git required) ==="
"$PY" - "$APP" "$STAGE" "${FILES[@]}" <<'PYCODE'
from pathlib import Path
import sys
live, stage, *files = sys.argv[1:]
def read(path):
    if not path.is_file():
        return None
    return path.read_bytes().replace(b"\r\n", b"\n")
for baseline in ("base0", "base1"):
    different = [
        name for name in files
        if read(Path(live) / name) != read(Path(stage) / baseline / name)
    ]
    if not different:
        print("Verified baseline:", baseline)
        break
else:
    sys.exit("STOP: live code differs from both expected baselines. No files changed.")
PYCODE

echo "=== Testing staged code before production changes ==="
"$PY" -m compileall -q "$STAGE/new/Core" "$STAGE/new/Api" "$STAGE/new/tools"
"$PY" -m json.tool "$STAGE/new/app_text.json" >/dev/null
mkdir -p "$STAGE/new/Tests/unit"
cp "$APP/Tests/unit/test_oracle_retrieval_quality.py" \
  "$STAGE/new/Tests/unit/test_oracle_retrieval_quality.py"
(
  cd "$STAGE/new"
  PYTHONPATH="$STAGE/new" "$PY" -m unittest discover \
    -s Tests/unit -p 'test_oracle_*quality.py' -v
  PYTHONPATH="$STAGE/new" "$PY" -m unittest discover \
    -s Tests/unit -p 'test_oracle_adaptive_voice.py' -v
)

echo "=== Backing up existing code ==="
mkdir -p /opt/omnissiah/backups
BACKUP=$(mktemp -d /opt/omnissiah/backups/adaptive-XXXXXXXX)
for f in ${FILES[@]}; do
  if [ -f "$APP/$f" ]; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -p "$APP/$f" "$BACKUP/$f"
  fi
done

echo "=== Installing and starting Oracle API ==="
APPLIED=1
for f in ${FILES[@]}; do
  install -D -m 644 "$STAGE/new/$f" "$APP/$f"
done
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

echo "=== Running real API acceptance tests ==="
"$PY" tools/oracle_voice_smoke.py
"$PY" tools/oracle_chronicle_smoke.py

echo "=== DEPLOYMENT AND SMOKE TESTS PASSED ==="
echo "Backup: $BACKUP"
