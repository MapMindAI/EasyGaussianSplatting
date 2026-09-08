#!/usr/bin/env bash
# Generate Python bindings from gsplat.proto. Run from the repository root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOC_FLAGS=(
  -I "${SCRIPT_DIR}"
  --python_out="${SCRIPT_DIR}"
  --grpc_python_out="${SCRIPT_DIR}"
)

# The images ship no protoc, so fall back to the copy grpcio-tools bundles,
# which is version-matched to the protobuf runtime it installed alongside.
if command -v protoc >/dev/null && command -v grpc_python_plugin >/dev/null; then
  protoc "${PROTOC_FLAGS[@]}" \
    --plugin="protoc-gen-grpc_python=$(command -v grpc_python_plugin)" \
    "${SCRIPT_DIR}/gsplat.proto"
else
  python3 -m grpc_tools.protoc "${PROTOC_FLAGS[@]}" "${SCRIPT_DIR}/gsplat.proto"
fi

sed -i 's/^import gsplat_pb2 as gsplat__pb2$/from . import gsplat_pb2 as gsplat__pb2/' \
  "${SCRIPT_DIR}/gsplat_pb2_grpc.py"
