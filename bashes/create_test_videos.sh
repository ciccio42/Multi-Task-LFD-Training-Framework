#!/bin/bash
# Generate demo/trajectory videos (H.264) for a test-results step folder.
# Usage: ./create_test_videos.sh /path/to/.../results_pick_place/run_1/step-199 [task_name]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STEP_PATH="${1:?Usage: $0 <step_path> [task_name]}"
TASK_NAME="${2:-pick_place}"

PYTHON_BIN="/mnt/beegfs/frosa/.conda/envs/multi_task_lfd_cuda_12_8/bin/python"

"$PYTHON_BIN" "$SCRIPT_DIR/create_test_videos.py" "$STEP_PATH" --task_name "$TASK_NAME"
