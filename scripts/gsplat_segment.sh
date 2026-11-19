#!/usr/bin/env bash
# Container: generate training masks in the active gsplat environment.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f gsplat_server/proto/gsplat_pb2.py ]; then
  bash gsplat_server/proto/build.sh
fi

python3 - "${1:?usage: $0 <workspace> <parameters.proto.txt> [parameters.proto.txt...]}" "${2:?parameters file required}" "${@:3}" <<'PY'
import sys
from pathlib import Path

from gsplat_server.parameters import load_parameters

directory = Path(sys.argv[1])
parameters = [load_parameters(path) for path in sys.argv[2:]]
mask_people = any(parameter.mask_people for parameter in parameters)
mask_sky = any(parameter.mask_sky for parameter in parameters)
if mask_people or mask_sky:
    from mapping.triton.segment_people import main

    main([
        str(directory / "images"), str(directory / "masks"),
        "--mask-people" if mask_people else "--no-mask-people",
        "--mask-sky" if mask_sky else "--no-mask-sky",
    ])
else:
    print("People and sky masking disabled; skipping segmentation")
PY
