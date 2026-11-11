#!/usr/bin/env bash
# OmnissiahCore: guarded Oracle update and real 70/30 acceptance.
# THIS CHANGES PRODUCTION ONLY AFTER BASELINE + UNIT TESTS PASS.
set -Eeuo pipefail
APP=/opt/omnissiah/app
RAW=https://raw.githubusercontent.com/Onoduchiagozie/OmnissiahCore
REV=407c63d7d13bd5a5e3f34b764b93d7dbea9f27c1
BASES=(
  3e7c0aba623ae613d98263c8b7f0e73bd9ef7191
  6130df296f83c0bb04c79335bb3709f8830fd40c
  fde0f1be3edbf16f1109c83707a4d1af3afd9c26
)
FILES=(
  Core/retriever.py Core/retrieval_quality.py Core/agent.py
  Core/prompt.py Core/response_style.py
  Api/models.py Api/routes/query_routes.py Api/services/runtime_service.py
  app_text.json
  Tests/unit/test_oracle_retrieval_quality.py
  Tests/unit/test_oracle_chronicle_quality.py
  Tests/unit/test_oracle_adaptive_voice.py
  Tests/unit/test_warhammer_eval_corpus.py
  Tests/data/warhammer_eval_70_30.json
  Tests/data/observed_test_results.json
  tools/oracle_chronicle_smoke.py
  tools/oracle_voice_smoke.py
  tools/oracle_benchmark_70_30.py
)
cd "$APP"
PY="$APP/.venv/bin/python"
test -x "$PY"
STAGE=$(mktemp -d)
BACKUP=""
CHANGED=0
cleanup() { rm -rf "$STAGE"; }
ready() { curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null 2>&1; }
wait_ready() {
  for _ in $(seq 1 90); do
    if ready; then return 0; fi
    sleep 2
  done
  return 1
}
rollback() {
  rc=$?
  trap - ERR
  if [ "$CHANGED" -eq 1 ]; then
    echo "=== ROLLBACK: restoring all previous files ==="
    for f in ${FILES[@]}; do
      if [ -f "$BACKUP/$f" ]; then
        mkdir -p "$(dirname "$APP/$f")"
        cp -p "$BACKUP/$f" "$APP/$f"
      else
        rm -f "$APP/$f"
      fi
    done
    if sudo systemctl restart omnissiah-api.service && wait_ready; then
      echo "ROLLBACK HEALTH: OK"
    else
      echo "ROLLBACK HEALTH: FAILED; inspect journalctl immediately" >&2
      sudo journalctl -u omnissiah-api.service -n 35 --no-pager >&2 || true
    fi
  fi
  echo "NOT DEPLOYED (exit $rc). Backup: $BACKUP" >&2
  exit "$rc"
}
trap cleanup EXIT
trap rollback ERR

echo "=== 0. Check existing Oracle API (no changes) ==="
sudo systemctl is-active --quiet omnissiah-api.service
ready
echo "Existing service online."

echo "=== 1. Stage immutable GitHub files ==="
for f in ${FILES[@]}; do
  mkdir -p "$STAGE/new/$(dirname "$f")"
  curl -fsSL --retry 2 "$RAW/$REV/$f" -o "$STAGE/new/$f"
done

echo "=== 2. Refuse unexpected local modifications ==="
for idx in "${!BASES[@]}"; do
  base="${BASES[$idx]}"
  for f in ${FILES[@]}; do
    mkdir -p "$STAGE/base$idx/$(dirname "$f")"
    status=$(curl -sSL --retry 2 -o "$STAGE/base$idx/$f" -w '%{http_code}' "$RAW/$base/$f")
    if [ "$status" = 404 ]; then
      rm -f "$STAGE/base$idx/$f"
    elif [ "$status" != 200 ]; then
      echo "Baseline fetch failed: HTTP $status $base/$f" >&2
      exit 1
    fi
  done
done
"$PY" - "$APP" "$STAGE" "${FILES[@]}" <<'PYCODE'
from pathlib import Path
import sys
live, stage, *files = sys.argv[1:]
def read(path):
    return path.read_bytes().replace(b"\r\n", b"\n") if path.is_file() else None
for baseline in ("base0", "base1", "base2"):
    diffs = [name for name in files
             if read(Path(live) / name) != read(Path(stage) / baseline / name)]
    if not diffs:
        print("Matches known deployment:", baseline, flush=True)
        break
else:
    sys.exit("STOP: Oracle files differ from all known baselines. No changes.")
PYCODE

echo "=== 3. Preflight syntax + all unit and 100 style tests ==="
"$PY" -m compileall -q "$STAGE/new/Core" "$STAGE/new/Api" "$STAGE/new/tools"
"$PY" -m json.tool "$STAGE/new/app_text.json" >/dev/null
(
  cd "$STAGE/new"
  PYTHONPATH="$STAGE/new" "$PY" -m unittest discover -s Tests/unit \
    -p 'test_oracle_*quality.py' -v
  PYTHONPATH="$STAGE/new" "$PY" -m unittest discover -s Tests/unit \
    -p 'test_oracle_adaptive_voice.py' -v
  PYTHONPATH="$STAGE/new" "$PY" -m unittest discover -s Tests/unit \
    -p 'test_warhammer_eval_corpus.py' -v
)

echo "=== 4. Back up production (including newly added files) ==="
mkdir -p /opt/omnissiah/backups
BACKUP=$(mktemp -d /opt/omnissiah/backups/eval-XXXXXXXX)
for f in ${FILES[@]}; do
  if [ -f "$APP/$f" ]; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    cp -p "$APP/$f" "$BACKUP/$f"
  fi
done
CHANGED=1
for f in ${FILES[@]}; do
  install -D -m 644 "$STAGE/new/$f" "$APP/$f"
done

echo "=== 5. Restart API; do not continue if unhealthy ==="
sudo systemctl restart omnissiah-api.service
wait_ready
curl -fsS --max-time 10 http://127.0.0.1:8000/health
echo

echo "=== 6. Run 100 real route checks, zero Gemini tokens ==="
"$PY" tools/oracle_benchmark_70_30.py --evidence-limit 0 \
  --out /tmp/oracle-voice-100.json

echo "=== 7. Test FINAL cross-encoder evidence for Horus/Davin ==="
"$PY" tools/oracle_chronicle_smoke.py

echo "=== 8. Sample final evidence across 7 narratives and 3 explanations ==="
"$PY" tools/oracle_benchmark_70_30.py --evidence-limit 10 \
  --out /tmp/oracle-evidence-70-30.json

echo "=== VERIFIED ON ORACLE (only if every preceding check passed) ==="
echo "Backup: $BACKUP"
echo "Reports: /tmp/oracle-voice-100.json, /tmp/oracle-evidence-70-30.json"
