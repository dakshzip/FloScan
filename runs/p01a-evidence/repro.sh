#!/usr/bin/env bash
# Reproduce the three P01 review defects through the user-facing CLI.
set -u
ROOT="$1"; WORK="$2"
cd "$ROOT"
INPUT="example input /1a8384c3f6"

echo "## Defect 1: fabricated complete success accepted by validate"
uv run python - "$WORK/fake-success.json" <<'PY'
import json, sys
e = json.load(open("runs/p01-evidence/live-lidar-1a8384c3f6/result.json"))
e["status"] = "ok"; e["exit_code"] = 0
e["coverage"]["contract_complete"] = True
for s in e["coverage"]["sections"].values():
    s["status"] = "available"
for st in e["stages"]:
    st["status"] = "ok"
json.dump(e, open(sys.argv[1], "w"), indent=2)
PY
echo "\$ uv run floscan validate fake-success.json"
uv run floscan validate "$WORK/fake-success.json" 2>&1 | sed "s#$WORK/##g"; echo "exit=${PIPESTATUS[0]}"

echo
echo "## Defect 1b: public_schema_export available while external schema unavailable"
uv run python - "$WORK/export-available.json" <<'PY'
import json, sys
e = json.load(open("runs/p01-evidence/live-lidar-1a8384c3f6/result.json"))
e["coverage"]["sections"]["public_schema_export"]["status"] = "available"
json.dump(e, open(sys.argv[1], "w"), indent=2)
PY
echo "\$ uv run floscan validate export-available.json"
uv run floscan validate "$WORK/export-available.json" 2>&1 | sed "s#$WORK/##g"; echo "exit=${PIPESTATUS[0]}"

echo
echo "## Defect 2: output parent is a regular file"
: > "$WORK/file-not-directory"
echo "\$ ./run.sh --input \"$INPUT\" --tier lidar --output file-not-directory/output --mode live"
./run.sh --input "$INPUT" --tier lidar --output "$WORK/file-not-directory/output" --mode live 2>&1 \
  | sed "s#$WORK/##g; s#$ROOT/##g" | tail -4; echo "exit=${PIPESTATUS[0]}"

echo
echo "## Defect 3: non-numeric source page count in gate registry"
sed 's/^    pages: 6$/    pages: not-a-number/' configs/gates.yaml > "$WORK/gates-bad-pages.yaml"
grep -n "pages: not-a-number" "$WORK/gates-bad-pages.yaml"
echo "\$ uv run floscan gates --gates gates-bad-pages.yaml"
uv run floscan gates --gates "$WORK/gates-bad-pages.yaml" 2>&1 | sed "s#$WORK/##g; s#$ROOT/##g" | tail -4; echo "exit=${PIPESTATUS[0]}"
