#!/usr/bin/env bash

set -euo pipefail

CONFIG_PATH="${1:-configs/tiny_experiment.yaml}"

echo "Running Tiny dataset training with config: ${CONFIG_PATH}"

CUDA_VISIBLE_DEVICES=0 \
python -m vit_scratch.experiment --config "${CONFIG_PATH}"
