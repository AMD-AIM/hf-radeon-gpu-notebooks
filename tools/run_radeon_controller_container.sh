#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 <container-suffix> <command> [args...]" >&2
  exit 2
fi

readonly CONTAINER_SUFFIX=$1
shift

: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"
: "${RADEON_CONTROLLER_IMAGE:?RADEON_CONTROLLER_IMAGE is required}"
: "${RADEON_CONTROLLER_NAME:?RADEON_CONTROLLER_NAME is required}"

if [[ ! "$CONTAINER_SUFFIX" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo "Invalid controller container suffix: $CONTAINER_SUFFIX" >&2
  exit 2
fi

environment_args=()
for variable in \
  HF_TOKEN \
  RADEON_API_TOKEN \
  RADEON_NOTEBOOK_API \
  RADEON_POD_IMAGE \
  RADEON_USER_NAME
do
  if [[ -v $variable ]]; then
    environment_args+=(--env "$variable")
  fi
done

exec docker run --rm \
  --name "${RADEON_CONTROLLER_NAME}-${CONTAINER_SUFFIX}" \
  --label ai.amd.hf-radeon-global-controller=true \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/controller-home \
  "${environment_args[@]}" \
  --volume "$GITHUB_WORKSPACE:/workspace" \
  --workdir /workspace \
  "$RADEON_CONTROLLER_IMAGE" \
  "$@"
