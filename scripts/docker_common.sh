#!/usr/bin/env bash
# Shared helpers for the host-side scripts that drive `docker run` themselves.
# Source it; don't execute it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/mapmindai/gaussiansplatting:latest}"
# Overridden by callers whose host exposes its GPU differently, e.g. Jetson.
read -r -a DOCKER_GPU_FLAGS <<< "${DOCKER_GPU_FLAGS:---gpus all}"
# Prefix for `docker run`; extra options and the image go after it. The volume
# persists downloaded PyTorch weights across runs. The shm size is a cap on a
# tmpfs, not a reservation: below roughly 8g the trainer's dataloader workers
# die with a bus error partway through a run.
DOCKER_RUN_FLAGS=(
  --rm "${DOCKER_GPU_FLAGS[@]}" --shm-size=8g
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
