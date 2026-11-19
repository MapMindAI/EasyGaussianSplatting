#!/usr/bin/env bash
# Container: generate training masks from JobParameters.
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/gsplat_env.sh
export PYTHONPATH="$PWD"

python3 - "${1:?usage: $0 <workspace> <parameters.proto.txt>}" "${2:?parameters file required}" <<'PY'
import subprocess
import sys
from pathlib import Path

from gsplat_server.parameters import load_parameters

directory = Path(sys.argv[1])
parameters = load_parameters(sys.argv[2])
subprocess.run([
    sys.executable, "-m", "mapping.triton.segment_people",
    str(directory / "images"), str(directory / "masks"),
    "--mask-people" if parameters.mask_people else "--no-mask-people",
    "--mask-sky" if parameters.mask_sky else "--no-mask-sky",
], check=True)
PY
