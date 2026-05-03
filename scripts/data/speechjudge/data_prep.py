#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2026 Wei Wang
#  MIT License (https://opensource.org/licenses/MIT)

"""Data preparation for SpeechJudge.

Produces two files:

* ``meta.jsonl`` — one path-free record per pair (canonical ``index``,
  *relative* wav filenames under ``<db>/<split>/``, raw preference label,
  derived MOS, precomputed durations). Safe to upload to the Hub: contains
  the human preference labels (the actual *meta*) without machine-specific
  paths.
* ``data_pairs.jsonl`` — the trainer-facing file with *absolute* audio paths,
  derived from ``meta.jsonl`` plus the local ``<db>/<split>/`` root.

Re-runs are cheap if ``meta.jsonl`` already exists: paths are re-expanded
against the local data dir without re-decoding audio. Audio extraction from
the upstream HF dataset is skipped per-pair when the wav files already exist
on disk.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator


PREFERENCE2MOS = {"A": 1.0, "B": -1.0, "Tie": 0.0}


def get_audio_duration(audio_path: Path) -> float:
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(audio_path.as_posix())
    meta = decoder.metadata
    if meta.duration_seconds_from_header is not None:
        return float(meta.duration_seconds_from_header)
    samples = decoder.get_all_samples()
    return float(samples.duration_seconds)


def build_meta_records(data: Path, split: str) -> list[dict]:
    """Iterate over the RMSnow/SpeechJudge-Data dataset, extracting wavs + meta."""
    import torchaudio
    from datasets import load_dataset
    from tqdm import tqdm

    # NOTE: the upstream dataset has occasional decoding failures, so we
    # iterate by index and tolerate per-sample errors.
    dataset = load_dataset("RMSnow/SpeechJudge-Data", split=split)
    records: list[dict] = []
    for i in tqdm(range(len(dataset)), desc=f"speechjudge/{split}"):
        try:
            sample = dataset[i]
        except Exception as e:
            logging.warning("Skipping sample %d due to loading error: %s", i, e)
            continue

        idx = sample["index"]
        fname_a, fname_b = f"{idx}_a.wav", f"{idx}_b.wav"
        wav_a, wav_b = data / fname_a, data / fname_b

        if not wav_a.exists():
            samples = sample["audioA"].get_all_samples()
            torchaudio.save(wav_a, samples.data, samples.sample_rate)
        if not wav_b.exists():
            samples = sample["audioB"].get_all_samples()
            torchaudio.save(wav_b, samples.data, samples.sample_rate)

        preference = sample["naturalness_label"]
        if preference not in PREFERENCE2MOS:
            logging.warning("Unknown preference label %r at index %s; skipping.", preference, idx)
            continue

        records.append({
            "index": idx,
            "filenames": [fname_a, fname_b],
            "preference": preference,
            "metrics": {"mos_overall": PREFERENCE2MOS[preference]},
            "durations": [get_audio_duration(wav_a), get_audio_duration(wav_b)],
        })
    return records


def load_meta(meta_path: Path) -> list[dict]:
    records: list[dict] = []
    with meta_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def meta_to_pairs(records: Iterable[dict], split_root: Path) -> Iterator[dict]:
    split_root = split_root.resolve()
    for rec in records:
        abs_paths = [(split_root / fname).as_posix() for fname in rec["filenames"]]
        yield {
            "audio_paths": abs_paths,
            "metrics": rec["metrics"],
            "durations": rec["durations"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Per-split data dir (e.g. <db>/train).")
    parser.add_argument("--split", type=str, required=True, choices=["train", "dev", "test", "other"])
    parser.add_argument("--out", type=Path, required=True, help="Output path for data_pairs.jsonl (absolute paths).")
    parser.add_argument("--meta", type=Path, default=None, help="Output path for meta.jsonl (path-free meta). Defaults to <out parent>/meta.jsonl.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    args.data.mkdir(parents=True, exist_ok=True)
    meta_path = args.meta if args.meta is not None else args.out.parent / "meta.jsonl"

    if meta_path.exists() and meta_path.stat().st_size > 0:
        logging.info("Reusing existing meta: %s (skipping audio decoding)", meta_path)
        records = load_meta(meta_path)
        # Sanity check that the audio files actually live where we expect.
        missing = [r["filenames"][0] for r in records[:5] if not (args.data / r["filenames"][0]).exists()]
        if missing:
            logging.warning(
                "meta.jsonl was reused but some audio files are missing under %s (e.g. %s). "
                "Run prepare.sh after downloading the audio, or delete meta.jsonl to force a full rebuild.",
                args.data, missing,
            )
    else:
        records = build_meta_records(args.data, args.split)
        n = write_jsonl(meta_path, records)
        logging.info("Wrote %d meta records to %s", n, meta_path)

    n = write_jsonl(args.out, meta_to_pairs(records, args.data))
    logging.info("Wrote %d pair records to %s", n, args.out)


if __name__ == "__main__":
    main()
