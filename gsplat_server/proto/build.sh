#!/usr/bin/env bash
# Generate Python bindings from gsplat.proto. Run from the repository root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOC_PLUGIN="$(command -v grpc_python_plugin)"

protoc -I "${SCRIPT_DIR}" \
  --python_out="${SCRIPT_DIR}" \
  --grpc_python_out="${SCRIPT_DIR}" \
  --plugin="protoc-gen-grpc_python=${PROTOC_PLUGIN}" \
  "${SCRIPT_DIR}/gsplat.proto"

sed -i 's/^import gsplat_pb2 as gsplat__pb2$/from . import gsplat_pb2 as gsplat__pb2/' \
  "${SCRIPT_DIR}/gsplat_pb2_grpc.py"
