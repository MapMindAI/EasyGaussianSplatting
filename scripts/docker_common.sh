#!/usr/bin/env bash
# Shared helpers for the host-side scripts that drive `docker run` themselves
# (run_pipeline.sh, run_gsplat.sh). Source it; don't execute it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/mapmindai/gaussiansplatting:latest}"
# Prefix for `docker run`; extra options and the image go after it. The volume
# persists downloaded PyTorch weights across runs.
DOCKER_RUN_FLAGS=(
  --rm --gpus all --shm-size=1g
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
