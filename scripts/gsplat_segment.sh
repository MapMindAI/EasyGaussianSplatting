#!/usr/bin/env bash
# Container: generate training masks from JobParameters.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gsplat_env.sh
export PYTHONPATH="$PWD"

if [ ! -f gsplat_server/proto/gsplat_pb2.py ]; then
  bash gsplat_server/proto/build.sh
fi

python3 - "${1:?usage: $0 <workspace> <parameters.proto.txt> [parameters.proto.txt...]}" "${2:?parameters file required}" "${@:3}" <<'PY'
import subprocess
import sys
from pathlib import Path

from gsplat_server.parameters import load_parameters

directory = Path(sys.argv[1])
parameters = [load_parameters(path) for path in sys.argv[2:]]
subprocess.run([
    sys.executable, "-m", "mapping.triton.segment_people",
    str(directory / "images"), str(directory / "masks"),
    "--mask-people" if any(parameter.mask_people for parameter in parameters) else "--no-mask-people",
    "--mask-sky" if any(parameter.mask_sky for parameter in parameters) else "--no-mask-sky",
], check=True)
PY
