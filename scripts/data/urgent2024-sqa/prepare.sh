#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Copyright 2026 Wei Wang
#  MIT License (https://opensource.org/licenses/MIT)

set -euo pipefail

db=$1

# Source path.sh if present (for env activation, WANDB_*); see path.sh.example.
if [ -f path.sh ]; then
    . path.sh
fi

echo "===== Start preparing [urgent2024-sqa] dataset ====="

# download is managed by huggingface datasets

mkdir -p ${db}/data data/urgent2024-sqa

if [ ! -f data/urgent2024-sqa/train/data.jsonl ]; then
    scripts/data/urgent2024-sqa/data_prep.py --data "${db}" --split "train" --out "${db}/data/train.csv"
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/urgent2024-sqa/train"
fi

if [ ! -f data/urgent2024-sqa/test/data.jsonl ]; then
    scripts/data/urgent2024-sqa/data_prep.py --data "${db}" --split "test" --out  "${db}/data/test.csv"
    scripts/data/csv2data.sh "${db}/data/test.csv" "data/urgent2024-sqa/test"
fi

echo "===== Finished preparing [urgent2024-sqa] dataset ====="