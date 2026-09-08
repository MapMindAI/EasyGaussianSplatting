#!/usr/bin/env bash
# Shared helpers for the host-side scripts that drive `docker run` themselves.
# Source it; don't execute it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/mapmindai/gaussiansplatting:latest}"
# Overridden by callers whose host exposes its GPU differently, e.g. Jetson.
read -r -a DOCKER_GPU_FLAGS <<< "${DOCKER_GPU_FLAGS:---gpus all}"
# Where the container reaches the Triton server the mapping stage infers
# against. The default resolves to the Docker host, for a server published on
# this machine's port 8011 (see doc/panorama_mapping.md). Triton's own gRPC
# port is 8001, but that one is too often already taken to default to.
TRITON_URL="${TRITON_URL:-host.docker.internal:8011}"
# Prefix for `docker run`; extra options and the image go after it. The volume
# persists downloaded PyTorch weights across runs. The shm size is a cap on a
# tmpfs, not a reservation: below roughly 8g the trainer's dataloader workers
# die with a bus error partway through a run. `host.docker.internal` is mapped
# explicitly because Docker on Linux, unlike Mac and Windows, does not.
DOCKER_RUN_FLAGS=(
  --rm "${DOCKER_GPU_FLAGS[@]}" --shm-size=8g
  --add-host host.docker.internal:host-gateway
  --mount "type=volume,source=easygaussiansplatting-torch-cache,target=/root/.cache/torch"
  -v "${REPO_ROOT}:/workspace" -w /workspace
)

# Echo a path relative to the repo checkout, which the container sees mounted
# at /workspace.
repo_relative_path() {
  local target="$1" absolute
  if [ -d "${target}" ]; then
    absolute="$(cd "${target}" && pwd)"
  else
    absolute="$(cd "$(dirname "${target}")" && pwd)/$(basename "${target}")"
  fi

  case "${absolute}" in
    "${REPO_ROOT}"/*) ;;
    *)
      echo "${target} must live under the repo checkout (${REPO_ROOT}) so the container can see it" >&2
      return 1
      ;;
  esac

  printf '%s\n' "${absolute#"${REPO_ROOT}"/}"
}
