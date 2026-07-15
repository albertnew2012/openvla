#!/usr/bin/env bash
# Build the OpenVLA image. Image name is derived from the repo folder (→ "openvla") and tagged with
# your username, matching the ubuntu2404 convention. Run from anywhere: `bash docker/build_image.sh`.
set -euo pipefail

# Get the full path to the script
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
DIR_NAME="$(basename "$(dirname "$SCRIPT_DIR")")"
IMG_NAME="$(echo "$DIR_NAME" | tr '[:upper:]' '[:lower:]')" # Convert to lowercase
USER_NAME=$(id -un)

echo "Building image: ${IMG_NAME}:${USER_NAME}  (from ${DIR_NAME}/docker/Dockerfile)"

docker build \
    --build-arg USER_NAME=$(id -un) \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    -t "$IMG_NAME:$USER_NAME" \
    "$SCRIPT_DIR"

# Skip the (slow) Flash-Attention build:
#   docker build --build-arg INSTALL_FLASH_ATTN=false ... "$SCRIPT_DIR"
# Force a clean rebuild:
#   docker build --no-cache ... "$SCRIPT_DIR"
