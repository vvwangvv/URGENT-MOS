#!/usr/bin/env python3

import json
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm

subjective_dimensions = [
    "mos_overall",
    "mos_naturalness",
    "mos_prosody",
    "mos_quality",
    "mos_expressiveness",
    "mos_fluency",
    "mos_paralinguistic",
]

METRICS = [
    "distill_mos",
    "dnsmos_ovrl",
    "estoi",
    "lps",
    "lsd",
    "mcd",
    "nisqa_mos",
    "pesqc2",
    "pesq",
    "sbert",
    "scoreq",
    "sdr",
    "sigmos_col",
    "sigmos_disc",
    "sigmos_loud",
    "sigmos_noise",
    "sigmos_ovrl",
    "sigmos_reverb",
    "sigmos_sig",
    "spksim",
    "utmos",
]

METRICS.extend(subjective_dimensions)


def csv2scps(csv_path, output_dir: Path):
    df = pd.read_csv(csv_path, dtype=str)

    # Per-utterance listener ratings for meta.jsonl: uttid -> { key: [(listener_id, rating), ...] }.

    # Mean score per (wav_path, dimension); unstack gives columns = dimension
    if "score" in df.columns:
        df["score"] = df["score"].astype(float)
        score_means = df.groupby(["wav_path", "dimension"])["score"].mean().unstack(level="dimension")

        for dimension in subjective_dimensions:
            if dimension in score_means.columns:
                df[dimension] = df["wav_path"].map(score_means[dimension])

    utt2listener_ratings, utt2listener_ratings_std = {}, {}
    if "listener_id" in df.columns:
        df["listener_id"] = df["listener_id"].fillna("").astype(str)
        for _, row in df.iterrows():
            uttid = f"{row['system_id']}:{row['reference_id']}"
            if uttid not in utt2listener_ratings:
                utt2listener_ratings[uttid] = {}
            if row["dimension"] not in utt2listener_ratings[uttid]:
                utt2listener_ratings[uttid][row["dimension"]] = []
            utt2listener_ratings[uttid][row["dimension"]].append((row["listener_id"], row["score"]))

    # NOTE: pstn and nisqa does not provide raw ratings but only standard deviation
    if "score_std" in df.columns:
        df["score_std"] = df["score_std"].astype(float)
        for _, row in df.iterrows():
            if math.isnan(row["score_std"]):
                continue
            uttid = f"{row['system_id']}:{row['reference_id']}"
            if uttid not in utt2listener_ratings_std:
                utt2listener_ratings_std[uttid] = {}
            utt2listener_ratings_std[uttid][row["dimension"]] = row["score_std"]

    unique_audios = set()
    utt2system, utt2reference, utt2audio_path, metric_to_utt2value = {}, {}, {}, {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating scp files"):
        audio_path = row["wav_path"]
        system_id, reference_id = row["system_id"], row["reference_id"]
        uttid = f"{system_id}:{reference_id}"
        if audio_path not in unique_audios:
            unique_audios.add(audio_path)
            assert uttid not in utt2system, f"duplicate reference_id: {uttid}"
            utt2system[uttid] = system_id
            utt2reference[uttid] = reference_id
            utt2audio_path[uttid] = row["wav_path"]

        # Merge metrics from every row so we get first non-empty value per metric per uttid.
        # NOTE: pandas reads empty CSV cells as float NaN even with dtype=str, so the
        # explicit pd.isna check is required (NaN is truthy under `not value`).
        for metric in METRICS:
            if metric not in row.index:
                continue

            value = row[metric]
            if value is None or value == "" or pd.isna(value):
                continue

            try:
                metric_value = round(float(value), 4)
            except (TypeError, ValueError):
                continue
            if math.isnan(metric_value):
                continue

            if metric not in metric_to_utt2value:
                metric_to_utt2value[metric] = {}
            if uttid not in metric_to_utt2value[metric]:
                metric_to_utt2value[metric][uttid] = metric_value

    uids = sorted(list(utt2audio_path.keys()))
    with (
        open(output_dir / "utt2system", "w") as utt2sys_scp,
        open(output_dir / "utt2reference", "w") as utt2reference_scp,
        open(output_dir / "wav.scp", "w") as wav_scp,
    ):
        for uid in uids:
            utt2sys_scp.write(f"{uid} {utt2system[uid]}\n")
            wav_scp.write(f"{uid} {utt2audio_path[uid]}\n")
            utt2reference_scp.write(f"{uid} {utt2reference[uid]}\n")

    # Write one .scp per metric in METRICS (missing metrics get an empty or partial file).
    for metric, utt2value in metric_to_utt2value.items():
        with open(output_dir / f"{metric}.scp", "w") as metric_scp:
            for uid in uids:
                if uid in utt2value:
                    metric_scp.write(f"{uid} {utt2value[uid]}\n")

    # meta.jsonl: one line per utterance, all subjective keys' listener->rating for that utterance.
    if utt2listener_ratings or utt2listener_ratings_std:
        with open(output_dir / "meta.jsonl", "w") as meta_jsonl:
            for uid in uids:
                rec = {"uid": uid}
                for dimension in subjective_dimensions:
                    if uid in utt2listener_ratings and dimension in utt2listener_ratings[uid]:
                        rec[dimension] = utt2listener_ratings[uid][dimension]
                    if uid in utt2listener_ratings_std and dimension in utt2listener_ratings_std[uid]:
                        rec[f"{dimension}_std"] = utt2listener_ratings_std[uid][dimension]
                if len(rec) > 1:
                    meta_jsonl.write(json.dumps(rec) + "\n")

    with open(output_dir / "data.jsonl", "w") as data_jsonl:
        for uid in uids:
            item = {
                "audio_path": utt2audio_path[uid],
                "uid": uid,
                "system_id": utt2system[uid],
                "reference_id": utt2reference[uid],
                "duration": None,
                "metrics": {},
            }
            for metric in METRICS:
                utt2val = metric_to_utt2value.get(metric)
                if utt2val and uid in utt2val:
                    item["metrics"][metric] = utt2val[uid]
            data_jsonl.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Make utt2system, utt2reference, wav.scp, mos.scp from csv")
    parser.add_argument("csv_path", type=Path, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=Path, help="Path to the output dir")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv2scps(args.csv_path, args.output_dir)
    print(f"Wrote scps to {args.output_dir}")
