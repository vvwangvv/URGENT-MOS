#!/usr/bin/env python3
"""Run URGENT-MOS inference on the urgent-challenge/vmc2026-track1-{dev,test} datasets.

Produces a submission-ready predictions.csv with header:

    sample_id,pred_score

ACR rows: absolute MOS clamped to [1, 5].
CCR rows: comparative CMOS clamped to [-3, +3]. Positive = audio_a is preferred.

Usage (from repo root):

    # Blind dev set (1,008 ACR + 2,520 CCR):
    python scripts/infer_vmc2026_track1.py --split dev --output predictions_dev.csv

    # Blind test set (4,032 ACR + 10,080 CCR):
    python scripts/infer_vmc2026_track1.py --split test --output predictions_test.csv

The schema matches the urgent-challenge/urgent2026-sqa dataset (same `acr`/`ccr`
configs, same audio columns), so the same checkpoint family used for URGENT 2026
Track 2 SQA can be reused here. Default checkpoint:
``urgent-challenge/urgent-mos-f1c1m5dcorpus``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from urgent_mos.api.infer import infer, infer_pairs
from urgent_mos.utils import load_model_from_checkpoint

DATASET_TEMPLATE = "urgent-challenge/vmc2026-track1-{split}"
METRIC = "mos_overall"

ACR_MIN, ACR_MAX = 1.0, 5.0
CCR_MIN, CCR_MAX = -3.0, 3.0


def _get_audio_tensor(row, key: str):
    """Extract 1D float waveform and sample rate from a dataset row (torchcodec or dict)."""
    audio_col = row[key]
    if hasattr(audio_col, "get_all_samples"):
        samples = audio_col.get_all_samples()
        waveform = samples.data.squeeze(0).float()
        sr = getattr(samples, "sample_rate", row.get("sample_rate", 16000))
    elif isinstance(audio_col, dict):
        waveform = torch.from_numpy(audio_col["array"].astype("float32"))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        sr = audio_col.get("sampling_rate", row.get("sample_rate", 16000))
    else:
        raise TypeError(f"Unexpected audio type for key {key!r}: {type(audio_col)}")
    return waveform, sr


def _to_scalar(score) -> float:
    if isinstance(score, (int, float)):
        return float(score)
    if hasattr(score, "item"):
        return float(score.item())
    return float(score[0])


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def run_acr(model, dataset, batch_frames: int | None):
    sample_ids: list[str] = []
    audios: list[torch.Tensor] = []
    sample_rates: list[int] = []
    for i in tqdm(range(len(dataset)), desc="Loading ACR"):
        row = dataset[i]
        sample_ids.append(row["sample_id"])
        wav, sr = _get_audio_tensor(row, "audio")
        audios.append(wav)
        sample_rates.append(sr)
    results = infer(
        model,
        audios,
        sample_rate=sample_rates,
        batch_frames=batch_frames,
        num_workers=0,
    )
    return [
        (sid, _clamp(_to_scalar(results[i][METRIC]), ACR_MIN, ACR_MAX))
        for i, sid in enumerate(sample_ids)
    ]


def run_ccr(model, dataset, batch_frames: int | None):
    sample_ids: list[str] = []
    audios_a: list[torch.Tensor] = []
    audios_b: list[torch.Tensor] = []
    sample_rates: list[tuple[int, int]] = []
    for i in tqdm(range(len(dataset)), desc="Loading CCR"):
        row = dataset[i]
        sample_ids.append(row["sample_id"])
        wa, sra = _get_audio_tensor(row, "audio_a")
        wb, srb = _get_audio_tensor(row, "audio_b")
        audios_a.append(wa)
        audios_b.append(wb)
        sample_rates.append((sra, srb))
    pairs = list(zip(audios_a, audios_b))
    results = infer_pairs(
        model,
        pairs,
        sample_rate=sample_rates,
        batch_frames=batch_frames,
        num_workers=0,
    )
    return [
        (sid, _clamp(_to_scalar(results[i][METRIC]), CCR_MIN, CCR_MAX))
        for i, sid in enumerate(sample_ids)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run URGENT-MOS on vmc2026-track1 (ACR + CCR) and write predictions.csv"
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="VMC2026 Track 1 split to run on (default: dev).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="urgent-challenge/urgent-mos-f1c1m5dcorpus",
        help="Path to model checkpoint (e.g. model.pt or HuggingFace repo id).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output predictions.csv path (default: predictions_<split>.csv).",
    )
    parser.add_argument(
        "--batch-frames",
        type=int,
        default=None,
        help="Max audio frames per batch (default: from config). Lower to avoid OOM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, only score the first N rows of each config (smoke testing).",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"predictions_{args.split}.csv"

    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.eval()

    dataset_id = DATASET_TEMPLATE.format(split=args.split)
    rows: list[tuple[str, float]] = []

    acr_ds = load_dataset(dataset_id, "acr", split=args.split)
    if args.limit is not None:
        acr_ds = acr_ds.select(range(min(args.limit, len(acr_ds))))
    rows.extend(run_acr(model, acr_ds, args.batch_frames))

    ccr_ds = load_dataset(dataset_id, "ccr", split=args.split)
    if args.limit is not None:
        ccr_ds = ccr_ds.select(range(min(args.limit, len(ccr_ds))))
    rows.extend(run_ccr(model, ccr_ds, args.batch_frames))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "pred_score"])
        for sid, score in rows:
            writer.writerow([sid, f"{score:.4f}"])
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
