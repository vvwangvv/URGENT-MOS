#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-path", required=True, type=Path, help="original csv file path.")
    parser.add_argument("--wavdir", required=True, type=Path, help="directory containing the waveform files.")
    parser.add_argument("--out", required=True, type=Path, help="output csv file path.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with open(args.original_path, newline="") as infile:
        num_lines = sum(1 for _ in infile) - 1

    with open(args.original_path, newline="") as infile, open(args.out, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(
            outfile,
            fieldnames=["wav_path", "system_id", "reference_id", "score", "dimension", "listener_id"],
        )
        writer.writeheader()

        for row in tqdm(reader, total=num_lines):
            if row["condition"].lower() == "ref":
                continue  # Skip ref rows

            wav_name = row["sample"]
            reference_id = wav_name.rsplit(".", 1)[0]
            system_id = row["condition"]
            wav_path = (args.wavdir / system_id / wav_name).resolve().as_posix()

            try:
                ratings = ast.literal_eval(row["rating_list"])
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Could not parse rating_list for {wav_name}: {row['rating_list']!r}") from e

            for score in ratings:
                writer.writerow(
                    {
                        "wav_path": wav_path,
                        "system_id": system_id,
                        "reference_id": reference_id,
                        "score": score,
                        "dimension": "mos_overall",
                        "listener_id": "",
                    }
                )


if __name__ == "__main__":
    main()
