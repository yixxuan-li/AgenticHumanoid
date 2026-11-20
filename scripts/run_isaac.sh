#!/usr/bin/env bash
set -euo pipefail
task_python="${ISAAC_PYTHON:-/home/lhy/miniconda3/envs/isaac51/bin/python}"
task_libstd="${ISAAC_LIBSTDCXX:-${task_python%/bin/python}/lib/libstdc++.so.6}"
if [[ -f "$task_libstd" ]]; then
  export LD_PRELOAD="$task_libstd${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export XDG_CACHE_HOME="${AGENTIC_HUMANOID_CACHE:-$PWD/outputs/isaaclab_runtime}"
export PYTHONUNBUFFERED=1
exec "$task_python" "$@"
