#!/usr/bin/env bash
set -euo pipefail

db=${1:-}

if [ -z "${db}" ]; then
    echo "Usage: $0 <rawdata-dir>" >&2
    exit 1
fi

# Datasets that are automatically downloaded and prepared.
auto_datasets=(
    tencent somos tmhint-qi chime-7-udase-eval tcd-voip ttsds2 pstn nisqa
    speecheval speechjudge
)
# Datasets that require manual processing after downloading.
manual_datasets=(bvcc bc19)

extra_datasets=(urgent2024-sqa urgent2025-sqa)

# datasets=("${auto_datasets[@]}" "${manual_datasets[@]}")
# use only auto datasets
datasets=("${auto_datasets[@]}")


# Bootstrap the per-utterance metadata sidecars (utt2dur, utt2system/reference,
# *.scp objective-metric files, meta.jsonl) so every dataset's collect_metrics.py
# can pick them up without re-running heavy NISQA/UTMOS/ScoreQ inference.
# Each dataset's prepare.sh still pulls its own raw audio archive separately.
META_REPO=${META_REPO:-urgent-challenge/vmc2026-track1-meta}
if [ ! -e "./data/.meta.done" ]; then
    mkdir -p ./data
    hf download --repo-type dataset "${META_REPO}" --local-dir ./data
    touch ./data/.meta.done
fi

mkdir -p "${db}"
for dset in "${datasets[@]}"; do
    "scripts/data/${dset}/prepare.sh" "${db}/${dset}"
done

# Build CMOS pair files for every non-train split.
for dset in "${datasets[@]}"; do
    if [ ! -d "data/${dset}" ]; then
        continue
    fi
    for f in $(find "data/${dset}" -name "data.jsonl"); do
        split_name=$(basename "$(dirname "${f}")")
        if [ "${split_name}" = "train" ]; then
            continue
        fi
        echo "Constructing pairs for ${dset}/${split_name}..."
        output_file="${f%.jsonl}_pairs.jsonl"
        if [ -f "${output_file}" ]; then
            echo "  pairs already constructed: ${output_file}"
            continue
        fi
        python scripts/data/construct_pairs.py "${f}" "${output_file}" \
            --pairing_scope auto --limit 100000
    done
done
