#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from torchcodec.decoders import AudioDecoder
from tqdm import tqdm


def get_duration(audio_path: str) -> float:
    decoder = AudioDecoder(audio_path)
    meta = decoder.metadata
    if meta.duration_seconds_from_header is not None:
        return float(meta.duration_seconds_from_header)
    return float(decoder.get_all_samples().duration_seconds)


def main():
    parser = argparse.ArgumentParser(description="Compute per-utterance duration from a wav.scp file.")
    parser.add_argument("--wav-scp", type=Path, required=True, help="Path to wav.scp")
    parser.add_argument("--out-scp", type=Path, required=True, help="Output utt2dur path")
    args = parser.parse_args()

    args.out_scp.parent.mkdir(parents=True, exist_ok=True)

    with open(args.wav_scp, "r") as f:
        num_lines = sum(1 for _ in f)

    with open(args.wav_scp, "r") as wav_scp, open(args.out_scp, "w") as f:
        for line in tqdm(wav_scp, total=num_lines, desc="Computing utt2dur"):
            line = line.rstrip("\n")
            if not line:
                continue
            # NOTE: split on first whitespace only, so wav paths containing spaces work.
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logging.warning("Skipping malformed wav.scp line: %r", line)
                continue
            uid, audio_path = parts
            try:
                duration = get_duration(audio_path)
            except Exception as e:
                logging.warning("Failed to decode %s (uid=%s): %s", audio_path, uid, e)
                continue
            f.write(f"{uid} {duration:.3f}\n")


if __name__ == "__main__":
    main()
