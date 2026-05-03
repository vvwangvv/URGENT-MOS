import argparse
import json
from pathlib import Path

from urgent_mos.utils import calculate_metrics

"""
Example jsonl line for ref:
{"sample_id": "sys0001-utt0001", "system_id": "sys0001", "metrics": {"mos": 3.5, "lps": 0.8}, "audio_path": "path/to/audio1.wav"}
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", type=Path, help="path to ref jsonl file")
    parser.add_argument("pred", type=Path, help="path to pred jsonl file")
    parser.add_argument("--ref-metric", type=str, default="mos_overall", help="metric key in ref file")
    parser.add_argument("--pred-metric", type=str, default="mos_overall", help="metric key in pred file")

    args = parser.parse_args()
    return args


def _load_refs(ref_path: Path, ref_metric: str) -> tuple[list[dict], dict[str, str]]:
    refs = []
    uid2system: dict[str, str] = {}
    with ref_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            ref_item = {
                "uid": item["uid"],
                "system_id": item["system_id"],
                "value": item["metrics"][ref_metric],
                "audio_path": item["audio_path"],
            }
            uid2system[ref_item["uid"]] = ref_item["system_id"]
            refs.append(ref_item)
    return refs, uid2system


def _load_preds(pred_path: Path, pred_metric: str, uid2system: dict[str, str]) -> list[dict]:
    preds = []
    with pred_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            uid = item["uid"]
            if uid not in uid2system:
                raise KeyError(f"Prediction uid not found in refs: {uid}")
            pred_item = {
                "uid": uid,
                "system_id": uid2system[uid],
                "value": item["metrics"][pred_metric],
                "audio_path": item["audio_path"],
            }
            preds.append(pred_item)
    return preds


def evaluate_files(
    ref_path: Path,
    pred_path: Path,
    ref_metric: str = "mos_overall",
    pred_metric: str = "mos_overall",
) -> dict[str, dict[str, float]]:
    refs, uid2system = _load_refs(ref_path, ref_metric)
    preds = _load_preds(pred_path, pred_metric, uid2system)
    results = calculate_metrics(preds, refs)
    return results


if __name__ == "__main__":
    args = get_args()
    results = evaluate_files(
        ref_path=args.ref,
        pred_path=args.pred,
        ref_metric=args.ref_metric,
        pred_metric=args.pred_metric,
    )

    print(json.dumps(results, indent=4))
