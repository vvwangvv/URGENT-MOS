# -*- coding: utf-8 -*-
#!/usr/bin/env bash
# Copyright 2026 Wei Wang
#  MIT License (https://opensource.org/licenses/MIT)
set -euo pipefail

# Source path.sh if present (for env activation, WANDB_*); see path.sh.example.
if [ -f path.sh ]; then
    . path.sh
fi

db=$1

name=$(basename ${db})
echo "===== Start preparing [${name}] dataset ====="


mkdir -p data/${name}

if [ ! -e data/${name}/train/data.jsonl ] && [ -f "${db}/data/train.csv" ]; then
    scripts/data/csv2data.sh "${db}/data/train.csv" "data/${name}/train"
fi

if [ ! -e data/${name}/dev/data.jsonl ] && [ -f "${db}/data/dev.csv" ]; then
    scripts/data/csv2data.sh "${db}/data/dev.csv" "data/${name}/dev"
fi

if [ ! -e data/${name}/test/data.jsonl ] && [ -f "${db}/data/test.csv" ]; then
    scripts/data/csv2data.sh "${db}/data/test.csv" "data/${name}/test"
fi


echo "===== Finished preparing [${name}] dataset ====="