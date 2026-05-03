import argparse
import itertools
import json
from pathlib import Path

from tqdm import tqdm


# python construct_pairs.py \
#     input data/somos/dev/data.jsonl \
#     output data_pairs/somos/dev.jsonl \
#     --pairing_scope reference


def construct_pairs(utt2reference, utt2mos, utt2audio_path, pairing_scope):

    def make_pair(utt1, utt2, scope):
        pair = {
            "audio_paths": (utt2audio_path[utt1], utt2audio_path[utt2]),
            "metrics": {
                "mos_overall": round(utt2mos[utt1] - utt2mos[utt2], 3),
            },
            "pairing_scope": scope,
        }
        return pair

    pair = None
    if pairing_scope in ["reference", "auto"]:
        reference2utts = {}
        for utt, reference in utt2reference.items():
            if reference not in reference2utts:
                reference2utts[reference] = []
            reference2utts[reference].append(utt)

        for reference, utts in reference2utts.items():
            for utt1 in utts:
                for utt2 in utts:
                    if utt1 == utt2:
                        continue
                    pair = make_pair(utt1, utt2, "reference")
                    yield pair

    if pair is None and pairing_scope in ["any", "auto"]:
        utts = list(utt2mos.keys())
        for i in tqdm(range(len(utts))):
            for j in range(i + 1, len(utts)):
                utt1, utt2 = utts[i], utts[j]
                if utt1 == utt2:
                    continue
                pair = make_pair(utt1, utt2, "any")
                yield pair


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct pairs of items from two input files.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to data.jsonl",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to the output pairs jsonl file",
    )
    parser.add_argument(
        "--pairing_scope",
        type=str,
        choices=["reference", "any", "auto"],
        default="auto",
        help="pairing scope: reference (sample with same reference), any (any sample with any reference), auto (try reference first, then any)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="maximum number of pairs to construct",
    )
    args = parser.parse_args()

    utt2reference, utt2mos, utt2audio_path = {}, {}, {}
    skipped = 0
    with open(args.input, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            uid = item["uid"]
            mos = item.get("metrics", {}).get("mos_overall")
            if mos is None:
                # Items without mos_overall can't form a CMOS pair; skip.
                skipped += 1
                continue
            utt2mos[uid] = mos
            utt2audio_path[uid] = item["audio_path"]
            utt2reference[uid] = item["reference_id"]
    if skipped:
        print(f"Skipped {skipped} item(s) with no mos_overall in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with open(args.output, "w") as f:
        for pair in itertools.islice(
            construct_pairs(utt2reference, utt2mos, utt2audio_path, args.pairing_scope), args.limit
        ):
            n_pairs += 1
            f.write(json.dumps(pair) + "\n")
    print(f"Constructed {n_pairs} pairs and saved to {args.output}.")
    if n_pairs == 0:
        args.output.unlink()
