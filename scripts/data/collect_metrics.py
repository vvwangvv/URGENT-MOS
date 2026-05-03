#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from csv2scps import METRICS


def read_scp(path: Path, key_type: Optional[Callable] = None, value_type: Optional[Callable] = None) -> dict[str, str]:
    if not path.exists():
        logging.warning(f"SCP file {path} does not exist, this can be normal for listeners.scp or moslist.scp")
        return {}
    result = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            uid, value = line.strip().split(maxsplit=1)
            if key_type is not None:
                uid = key_type(uid)
            if value_type is not None:
                value = value_type(value)
            result[uid] = value
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert CSV to JSONL")
    parser.add_argument("data", type=Path, help="Path to data dir")
    parser.add_argument("jsonl_path", type=Path, help="Path to output JSONL file")

    args = parser.parse_args()

    metric_scps = [path for path in args.data.glob("*.scp") if path.stem in METRICS]

    utt2audio_path = read_scp(args.data / "wav.scp")
    utt2system = read_scp(args.data / "utt2system")
    utt2reference = read_scp(args.data / "utt2reference")
    utt2dur = read_scp(args.data / "utt2dur", value_type=float)

    utt2meta = {}
    if (args.data / "meta.jsonl").exists():
        with open(args.data / "meta.jsonl", "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                uid = item.pop("uid")
                utt2meta[uid] = item

    metric_to_utt2score = {}
    for metric_scp in metric_scps:
        metric = metric_scp.stem
        metric_to_utt2score[metric] = read_scp(metric_scp, value_type=lambda x: round(float(x), 4))

    items = []
    for uid in sorted(utt2audio_path.keys()):
        audio_path = utt2audio_path[uid]
        item = {
            "audio_path": audio_path,
            "uid": uid,
            "system_id": utt2system[uid],
            "reference_id": utt2reference[uid],
            "duration": utt2dur[uid],
            "metrics": {},
            "meta": utt2meta.get(uid, {}),
        }
        for metric, utt2score in metric_to_utt2score.items():
            if uid in utt2score:
                item["metrics"][metric] = utt2score[uid]
        items.append(item)

    with open(args.jsonl_path, "w") as jsonl_file:
        for item in items:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")
