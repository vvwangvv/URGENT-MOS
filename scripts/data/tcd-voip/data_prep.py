#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data preparation for TCD-VOIP."""

import argparse
import csv
import os
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert TCD-VOIP MOS xlsx to flat CSV.")
    parser.add_argument("--xlsx", required=True, type=str, help="Path to the input MOS xlsx file.")
    parser.add_argument("--wavdir", required=True, type=str, help="Directory containing the wav files.")
    parser.add_argument("--out", required=True, type=str, help="Path to the output CSV file.")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.xlsx, sheet_name="Subjective Test Scores")

    df["wav_path"] = df["Filename"].apply(lambda x: os.path.join(args.wavdir, x))
    df["system_id"] = df["ConditionID"].astype(str).str.zfill(3)
    df["reference_id"] = df["Filename"].apply(lambda x: os.path.splitext(x)[0][5:])

    fieldnames = ["wav_path", "system_id", "reference_id", "score", "dimension", "listener_id"]
    with open(args.out, "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for _, row in df.iterrows():
            for listener_id in range(1, 25):
                if listener_id not in row.index:
                    continue
                writer.writerow(
                    {
                        "wav_path": row["wav_path"],
                        "system_id": row["system_id"],
                        "reference_id": row["reference_id"],
                        "score": float(row[listener_id]),
                        "dimension": "mos_overall",
                        "listener_id": str(listener_id),
                    }
                )


if __name__ == "__main__":
    main()
