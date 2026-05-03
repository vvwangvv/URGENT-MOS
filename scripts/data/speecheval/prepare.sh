#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Copyright 2026 Wei Wang
#  MIT License (https://opensource.org/licenses/MIT)
set -euo pipefail

db=$1

echo "===== Start preparing [speecheval] dataset ====="

mkdir -p data/speecheval

# download dataset
if [ ! -e ${db}/download.done ]; then
    mkdir -p ${db}
    pushd ${db}
    # wget https://zenodo.org/records/7378801/files/somos.zip
    hf download --local-dir rawdata --repo-type dataset Hui519/SpeechEval
    hf download --local-dir preference --repo-type dataset vvwangvv/SpeechEval-Preference
    popd
    echo "Successfully finished download."
    touch ${db}/download.done
else
    echo "Already exists. Skip download."
fi


if [ ! -f data/speecheval/train/data_pairs.jsonl ]; then
    scripts/data/speecheval/data_prep.py --data "${db}" --split "train" --out "data/speecheval/train/data_pairs.jsonl" --preference "${db}/preference/train.jsonl"
fi

if [ ! -f data/speecheval/dev/data_pairs.jsonl ]; then
    scripts/data/speecheval/data_prep.py --data "${db}" --split "validation" --out  "data/speecheval/dev/data_pairs.jsonl" --preference "${db}/preference/validation.jsonl"
fi

if [ ! -f data/speecheval/test/data_pairs.jsonl ]; then
    scripts/data/speecheval/data_prep.py --data "${db}" --split "test" --out  "data/speecheval/test/data_pairs.jsonl" --preference "${db}/preference/test.jsonl"
fi

echo "===== Finished preparing [speecheval] dataset ====="