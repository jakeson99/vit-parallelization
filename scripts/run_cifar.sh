#!/usr/bin/env bash

set -euo pipefail

# Default config path can be overridden with CONFIG env or first arg
CONFIG_PATH="${1:-configs/experiment.yaml}"

echo "Running CIFAR training with config: ${CONFIG_PATH}"

CUDA_VISIBLE_DEVICES=0 \
python -m vit_scratch.experiment --config "${CONFIG_PATH}"
