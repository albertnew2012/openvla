#!/usr/bin/env bash
# Create + enter the OpenVLA container. Mounts your home (so the repo, ~/.cache/huggingface model
# weights, and edits are shared), maps the GPU, and registers the editable `openvla`/`prismatic`
# package against the mounted repo on entry. Mirrors the ubuntu2404 create_container.sh.
set -euo pipefail

USER_NAME=$(id -un)

# Get the full path to the script
DOCKERFILE_PATH="$(realpath "$0")"
DOCKER_DIR="$(dirname "$DOCKERFILE_PATH")"
REPO_PATH="$(dirname "$DOCKER_DIR")"
REPO_NAME="$(basename "$REPO_PATH")"
IMG_NAME="$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]')" # Convert to lowercase

# If a container with this name already exists, just re-enter it.
if docker ps -a --format '{{.Names}}' | grep -qx "${IMG_NAME}_${USER_NAME}"; then
    echo "Container ${IMG_NAME}_${USER_NAME} exists — attaching (start if stopped)."
    docker start "${IMG_NAME}_${USER_NAME}" >/dev/null
    exec docker exec -it -w "$REPO_PATH" "${IMG_NAME}_${USER_NAME}" bash
fi

docker run \
    -it \
    --env DISPLAY=$DISPLAY \
    --gpus all \
    --network=host \
    --shm-size=16g \
    --privileged=true \
    --name ${IMG_NAME}_${USER_NAME} \
    -v /home/$USER:/home/$USER_NAME \
    --mount type=volume,dst=/home/$USER_NAME/.local,volume-nocopy \
    -v /etc/localtime:/etc/localtime:ro \
    -w $REPO_PATH \
    $IMG_NAME:${USER_NAME} \
    bash -lc "sudo mkdir -p /home/$USER_NAME/.local \
      && sudo chown -R \$(id -u):\$(id -g) /home/$USER_NAME/.local \
      && echo '[entry] registering editable openvla/prismatic against the mounted repo...' \
      && python3 -m pip install -e . --no-deps -q 2>/dev/null || true \
      && exec bash"
