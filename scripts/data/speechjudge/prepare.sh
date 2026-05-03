#!/usr/bin/env bash
set -e

db=$1

echo "===== Start preparing [speechjudge] dataset ====="

mkdir -p data/speechjudge


if [ ! -f data/speechjudge/train/data_pairs.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}/train" --split "train" --out "data/speechjudge/train/data_pairs.jsonl"
fi

if [ ! -f data/speechjudge/dev/data_pairs.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}/dev" --split "dev" --out  "data/speechjudge/dev/data_pairs.jsonl"
fi

if [ ! -f data/speechjudge/test/data_pairs.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}/test" --split "test" --out  "data/speechjudge/test/data_pairs.jsonl"
fi

if [ ! -f data/speechjudge/other/data_pairs.jsonl ]; then
    scripts/data/speechjudge/data_prep.py --data "${db}/other" --split "other" --out  "data/speechjudge/other/data_pairs.jsonl"
fi

echo "===== Finished preparing [speechjudge] dataset ====="