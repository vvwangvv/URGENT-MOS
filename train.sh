#!/usr/bin/env bash
# Single-machine, multi-GPU training launcher.
# Number of processes is taken from CUDA_VISIBLE_DEVICES if set, else from
# `nvidia-smi -L`. For any non-trivial setup (multi-node, FSDP, DeepSpeed, ...)
# call `accelerate launch` directly.
set -e

. path.sh

config_name=${1:-f1c1m5_d_corpus}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    num_processes=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    num_processes=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
: "${num_processes:=1}"
[ "$num_processes" -lt 1 ] && num_processes=1

accelerate launch --mixed_precision=bf16 --num_processes="${num_processes}" \
    -m urgent_mos.train \
        --config-name="${config_name}" \
        exp_dir="exp/${config_name}"
