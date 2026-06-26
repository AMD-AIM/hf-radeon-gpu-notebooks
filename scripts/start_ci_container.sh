#!/usr/bin/env bash
# Start the persistent Radeon notebook-CI dev container on GPU 2 + 3.
#
# All model weights and the repo are BIND-MOUNTED, so the container itself holds
# no state — it is safe to stop / remove / recreate at any time.
#
#   ./scripts/start_ci_container.sh              # create (or re-start if exists)
#   ./scripts/start_ci_container.sh --recreate   # force rebuild from scratch
#   docker exec -it hf_radeon_ci bash            # enter the container
#
# Verified host: wx-ms-w7900d-0033  (GPU 2 -> renderD130, GPU 3 -> renderD131)
set -euo pipefail

NAME="${NAME:-hf_radeon_ci}"
IMAGE="${IMAGE:-huaggingface_for_amd_radeon:latest}"
REPO="${REPO:-/home/zihaomu/big_card/notebook_polish/hf-radeon-gpu-notebooks}"
VIDEO_GID="${VIDEO_GID:-44}"
RENDER_GID="${RENDER_GID:-993}"
# GPU 2 -> renderD130, GPU 3 -> renderD131
GPU_RENDER_NODES=( /dev/dri/renderD130 /dev/dri/renderD131 )

RECREATE=0
[[ "${1:-}" == "--recreate" ]] && RECREATE=1

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  if [[ "$RECREATE" == "1" ]]; then
    echo ">> removing existing container '$NAME' ..."
    docker rm -f "$NAME"
  else
    echo ">> container '$NAME' already exists; starting it (pass --recreate to rebuild)."
    docker start "$NAME" >/dev/null
    echo ">> ready. Enter with:  docker exec -it $NAME bash"
    exit 0
  fi
fi

DEV_ARGS=( --device=/dev/kfd )
for n in "${GPU_RENDER_NODES[@]}"; do DEV_ARGS+=( --device="$n" ); done

echo ">> creating persistent container '$NAME' on GPU 2 + 3 ..."
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  "${DEV_ARGS[@]}" \
  --group-add "$VIDEO_GID" --group-add "$RENDER_GID" \
  --security-opt seccomp=unconfined \
  --ipc=host --shm-size 32g \
  -v /disk/ssd1:/disk/ssd1 \
  -v /disk/ssd2:/disk/ssd2 \
  -v "$REPO":/workspace \
  -w /workspace \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint /bin/bash \
  "$IMAGE" \
  -lc 'sleep infinity'

echo ">> created. Verifying GPU visibility inside the container ..."
docker exec "$NAME" python3 -c "import torch; print('torch', torch.__version__, '| visible GPUs', torch.cuda.device_count())"
echo ">> done. Enter with:  docker exec -it $NAME bash"
