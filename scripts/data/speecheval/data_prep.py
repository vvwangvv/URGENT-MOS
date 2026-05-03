#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2026 Wei Wang
#  MIT License (https://opensource.org/licenses/MIT)
"""Data preparation for SpeechEval (CompareEval pairs).

Produces two files:

* ``meta.jsonl`` — one path-free record per pair (canonical key, *relative*
  audio paths under ``<db>/rawdata/``, raw preference label, derived MOS,
  precomputed durations). Safe to upload to the Hub: contains the human
  preference labels (the actual *meta*) without any machine-specific paths.
* ``data_pairs.jsonl`` — the trainer-facing file with *absolute* audio paths,
  derived from ``meta.jsonl`` plus the local ``<db>/rawdata/`` root.

Re-runs are cheap: if ``meta.jsonl`` already exists for the split, the script
re-expands paths against the local ``<db>/rawdata/`` and writes
``data_pairs.jsonl`` without touching the audio (no duration decoding,
no HF dataset traversal).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm


PREFERENCE2MOS = {"A": 1.0, "B": -1.0, "tie": 0.0}


def get_audio_duration(audio_path: Path) -> float:
    from torchcodec.decoders import AudioDecoder

    decoder = AudioDecoder(audio_path.as_posix())
    meta = decoder.metadata
    if meta.duration_seconds_from_header is not None:
        return float(meta.duration_seconds_from_header)
    samples = decoder.get_all_samples()
    return float(samples.duration_seconds)


def build_meta_records(
    data: Path,
    split: str,
    key2preference: dict[str, str],
) -> Iterator[dict]:
    """Yield path-free meta records by joining the SpeechEval HF dataset with the preference labels."""
    from datasets import load_dataset

    dataset = load_dataset((data / "rawdata").as_posix(), split=split)
    relpath2dur: dict[str, float] = {}
    for sample in tqdm(dataset, desc=f"speecheval/{split}"):
        if sample["task"] != "CompareEval":
            continue
        if sample["key"] not in key2preference:
            continue
        preference = key2preference[sample["key"]]
        if preference not in PREFERENCE2MOS:
            logging.warning("Unknown preference label %r for key %s; skipping.", preference, sample["key"])
            continue
        relpath_1 = sample["path"]
        relpath_2 = sample["path_B"]
        abs_1 = data / "rawdata" / relpath_1
        abs_2 = data / "rawdata" / relpath_2
        if not (abs_1.exists() and abs_2.exists()):
            logging.warning("Missing audio for key %s: %s or %s", sample["key"], abs_1, abs_2)
            continue
        if relpath_1 not in relpath2dur:
            relpath2dur[relpath_1] = get_audio_duration(abs_1)
        if relpath_2 not in relpath2dur:
            relpath2dur[relpath_2] = get_audio_duration(abs_2)
        yield {
            "key": sample["key"],
            "audio_relpaths": [relpath_1, relpath_2],
            "preference": preference,
            "metrics": {"mos_overall": PREFERENCE2MOS[preference]},
            "durations": [relpath2dur[relpath_1], relpath2dur[relpath_2]],
        }


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


def meta_to_pairs(records: Iterable[dict], rawdata_root: Path) -> Iterator[dict]:
    rawdata_root = rawdata_root.resolve()
    for rec in records:
        abs_paths = [(rawdata_root / rel).as_posix() for rel in rec["audio_relpaths"]]
        yield {
            "audio_paths": abs_paths,
            "metrics": rec["metrics"],
            "durations": rec["durations"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="SpeechEval root dir (must contain rawdata/).")
    parser.add_argument("--preference", type=Path, required=True, help="Path to the preference jsonl for this split.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, required=True, help="Output path for data_pairs.jsonl (absolute paths).")
    parser.add_argument("--meta", type=Path, default=None, help="Output path for meta.jsonl (path-free meta). Defaults to <out parent>/meta.jsonl.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    meta_path = args.meta if args.meta is not None else args.out.parent / "meta.jsonl"

    if meta_path.exists() and meta_path.stat().st_size > 0:
        logging.info("Reusing existing meta: %s (skipping audio decoding)", meta_path)
        records = load_meta(meta_path)
    else:
        key2preference = {}
        with args.preference.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                key2preference[item["key"]] = item["preference"]
        records = list(build_meta_records(args.data, args.split, key2preference))
        n = write_jsonl(meta_path, records)
        logging.info("Wrote %d meta records to %s", n, meta_path)

    n = write_jsonl(args.out, meta_to_pairs(records, args.data / "rawdata"))
    logging.info("Wrote %d pair records to %s", n, args.out)


if __name__ == "__main__":
    main()
