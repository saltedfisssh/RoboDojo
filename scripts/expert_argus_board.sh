#!/usr/bin/env bash
# Publish an exported expert dataset on an Argus board (planner labels, no VLM call).
#   bash scripts/expert_argus_board.sh data/expert/lerobot/stack_blocks [BOARD_NAME]
# Then serve it:  cd $ARGUS && .venv/bin/python -m board serve --board data/boards/<BOARD_NAME> --clips data/boards/<BOARD_NAME>/clips --port 9998
set -euo pipefail
DATASET="$(cd "$1" && pwd)"
NAME="$(basename "$DATASET")"
BOARD="${2:-robodojo_expert}"
ARGUS="${ARGUS_ROOT:?Set ARGUS_ROOT to your own Argus checkout; board publishing is optional.}"
PY="$ARGUS/.venv/bin/python"
EPISODES="$ARGUS/data/episodes/robodojo_expert/$NAME"
RUN="$ARGUS/data/runs/robodojo_expert/$NAME-planner"
BOARD_DIR="$ARGUS/data/boards/$BOARD"
cd "$ARGUS"
"$PY" -m prepare lerobot prepare --root "$DATASET" --rig teleop_arms --out "$EPISODES" --dataset "robodojo_expert" --force
mkdir -p "$RUN/out" "$BOARD_DIR"
"$PY" - "$DATASET" "$EPISODES" "$RUN" <<'EOF'
import json, sys, datetime
from pathlib import Path
dataset, episodes, run = map(Path, sys.argv[1:])
n = 0
for f in sorted((dataset / "meta/argus").glob("episode_*.json")):
    label = json.loads(f.read_text())
    label["episode_dir"] = str(episodes / f.stem)
    (run / "out" / f.name).write_text(json.dumps(label, indent=1))
    n += 1
now = datetime.datetime.now().isoformat(timespec="seconds")
(run / "run.json").write_text(json.dumps({
    "run_id": run.name, "kind": "full", "dataset": "robodojo_expert", "slice": str(episodes),
    "code": "planner", "code_dirty": False, "started_at": now, "finished_at": now, "status": "done",
    "episodes_done": n, "episodes_parsed": n, "episodes_failed": 0, "cost_usd": 0.0,
    "note": "labels generated from simulator ground truth by src/expert/argus.py",
}, indent=1))
print(f"{n} labels -> {run}")
EOF
MANIFEST="$BOARD_DIR/manifest.json"
"$PY" - "$MANIFEST" "$NAME" "$RUN" "$EPISODES" <<'EOF'
import json, sys
from pathlib import Path
manifest, name, run, episodes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
m = json.loads(Path(manifest).read_text()) if Path(manifest).exists() else {"board": "RoboDojo expert", "datasets": []}
m["datasets"] = [d for d in m["datasets"] if d["dataset"] != f"expert_{name}"]
m["datasets"].append({"dataset": f"expert_{name}", "run": run, "episodes": episodes, "file_prefix": f"{name}_"})
Path(manifest).write_text(json.dumps(m, indent=2))
EOF
"$PY" -m board build "$BOARD_DIR"
"$PY" -m board clips --episodes "$EPISODES" --out "$BOARD_DIR/clips"
echo "board: $BOARD_DIR"
