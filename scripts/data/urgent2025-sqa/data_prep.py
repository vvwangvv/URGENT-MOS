#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2026 Wei Wang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for Urgent2025-SQA."""

import argparse
from pathlib import Path

import pandas as pd
import torchaudio
from datasets import load_dataset
from tqdm import tqdm


def prepare_data(data: Path, split: str):
    if split == "train":
        # NOTE: The blind test set has both MOS-labeled and unlabeled samples
        # enhanced from the same set of noisy base recordings. Using unlabeled
        # samples for training is optional: they don't overlap with labeled
        # test data but share the same source, so opinions on fairness may differ.
        phases = ["validation", "nonblind_test"]  # + ["blind_test"]
    elif split == "test":
        phases = ["blind_test_mos"]

    items = []
    for phase in phases:
        for sample in tqdm(load_dataset("urgent-challenge/urgent2025-sqa", split=phase)):
            sample["reference_id"] = sample["sample_id"]
            if sample["mos"] is not None:
                sample["score"] = sample["mos"]
                sample["dimension"] = "mos_overall"
            del sample["mos"]
            del sample["sample_id"]
            submission_id = sample["system_id"].rsplit("_", 1)[1]
            fileid = sample["reference_id"].rsplit("_", 1)[1]
            wav_file = data / phase / submission_id / f"{fileid}.flac"
            wav_file.parent.mkdir(parents=True, exist_ok=True)
            if not wav_file.exists():
                samples = sample["audio"].get_all_samples()
                torchaudio.save(wav_file, samples.data, samples.sample_rate)
            del sample["audio"]
            sample["wav_path"] = wav_file.absolute().as_posix()
            sample = {k: v for k, v in sample.items() if v is not None}
            items.append(sample)
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = prepare_data(args.data, args.split)
    df = pd.DataFrame(items)
    df.to_csv(args.out, index=False)
