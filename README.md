# UrgentMOS — VMC 2026 Track 1 Baseline

[![arXiv](https://img.shields.io/badge/arXiv-2601.18438-b31b1b.svg)](https://arxiv.org/abs/2601.18438)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-models-yellow)](https://huggingface.co/urgent-challenge)

Official baseline for the
[VoiceMOS Challenge 2026 — Track 1](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026),
implementing
[arXiv:2601.18438](https://arxiv.org/abs/2601.18438)). The model jointly
predicts:

- **Absolute MOS** for ACR rows (1–5 scale).
- **Comparative MOS / CMOS** for CCR rows (signed pairwise preference,
  −3…+3, positive ⇒ A is preferred over B).

---

## Install

```bash
git clone https://github.com/vvwangvv/URGENT-MOS.git
cd URGENT-MOS

# inference only
pip install -e .

# add training-time deps
pip install -e ".[train]"
```

---

## Quickstart — submit a baseline

Generate `predictions.csv` for dev / test using the published checkpoint
[`urgent-challenge/urgent-mos-f1c1m5dcorpus`](https://huggingface.co/urgent-challenge/urgent-mos-f1c1m5dcorpus):

```bash
# dev: 1,008 ACR + 2,520 CCR = 3,528 rows
python scripts/infer_vmc2026_track1.py --split dev  --output predictions_dev.csv

# test: 4,032 ACR + 10,080 CCR = 14,112 rows
python scripts/infer_vmc2026_track1.py --split test --output predictions_test.csv
```

Output format (header + one row per `sample_id`):

```text
sample_id,pred_score
vmc2026-track1-dev-acr_0,4.4348
vmc2026-track1-dev-ccr_0,-0.5210
...
```

ACR rows are clamped to `[1, 5]`, CCR rows to `[-3, +3]`. Use a different
checkpoint (HF repo id or local `.pt`) with `--checkpoint`, lower
`--batch-frames` for OOM, or `--limit N` for a smoke test.

---

## Inference API

```python
from urgent_mos import infer, infer_pairs, load_model_from_checkpoint

model = load_model_from_checkpoint(
    "urgent-challenge/urgent-mos-f1c1m5dcorpus", device="cuda"
)

# Absolute MOS — one score per audio
print(infer(model, ["audio.wav"])[0]["mos_overall"])

# Comparative CMOS — positive ⇒ A is preferred over B
print(infer_pairs(model, [("a.wav", "b.wav")])[0]["mos_overall"])
```

`audio_inputs` may be a path, a numpy array, or a 1-D `torch.Tensor`; pass
`sample_rate=` for non-path inputs. `scripts/infer_vmc2026_track1.py` is a
thin wrapper over these two calls that also writes the submission CSV.

---

## Train from scratch

### 1. Prepare the data

```bash
./scripts/data/prepare.sh /path/to/raw_audio_root
```

This downloads the precomputed metric sidecars from
[`urgent-challenge/vmc2026-track1-meta`](https://huggingface.co/datasets/urgent-challenge/vmc2026-track1-meta)
into `data/`, then runs each per-corpus `prepare.sh` under `scripts/data/`
to produce a `data.jsonl` (and `data_pairs.jsonl` where applicable) per
dataset. 

The two license-gated corpora (BC19, BVCC) require a manual approval after downloading;
URGENT 2024 SQA, URGENT 2025 SQA are time-consuming to prepare.
These four datasets are by default excluded from data prepartion.

### 2. Activate your environment

```bash
cp path.sh.example path.sh    # one-time: edit to activate your env + WANDB_*
. path.sh
```

### 3. Train

```bash
./train.sh                                    # default: configs/f1c1m5_d_corpus.yaml
./train.sh f1c1m5_d_corpus                    # explicit config name (no .yaml suffix)
```

`train.sh` is a tiny single-machine wrapper around `accelerate launch`
(bf16). The number of processes is auto-detected from `CUDA_VISIBLE_DEVICES`
(falling back to `nvidia-smi -L` if it is unset), so
`CUDA_VISIBLE_DEVICES=0,1,2,3 ./train.sh` will use 4 GPUs on its own.
For any non-trivial setup (multi-node) call
`accelerate launch` directly — see the
[🤗 Accelerate documentation](https://huggingface.co/docs/accelerate/) for
all available options.

Any field can be overridden inline via
Hydra, e.g. `./train.sh f1c1m5_d_corpus dataloader.num_workers=8 trainer.learning_rate=2.4e-4`.

Training logs to W&B by default — disable with `WANDB_MODE=disabled`.

### 4. Validate your checkpoint

```bash
python scripts/infer_vmc2026_track1.py \
    --split dev \
    --checkpoint exp/f1c1m5_d_corpus/model_<step>.pt \
    --output predictions_dev.my_run.csv
```

---

## Acknowledgements

This baseline is inspired by, and shares design ideas with:

- [`unilight/sheet`](https://github.com/unilight/sheet) — Speech Human
  Evaluation Estimation Toolkit (SHEET) and the MOS-Bench dataset suite.
- [`urgent-challenge/urgent2026_challenge_track2`](https://github.com/urgent-challenge/urgent2026_challenge_track2)
  — official baseline for the URGENT 2026 Challenge Track 2 (Universa-Ext).

---

## Citation

```bibtex
@misc{wang2026urgentmos,
  title         = {UrgentMOS: Unified Multi-Metric and Preference Learning for Robust Speech Quality Assessment},
  author        = {Wang, Wei and Zhang, Wangyou and Li, Chenda and Wang, Jiahe and Cornell, Samuele and Sach, Marvin and Saijo, Kohei and Fu, Yihui and Ni, Zhaoheng and Han, Bing and Gong, Xun and Bi, Mengxiao and Fingscheidt, Tim and Watanabe, Shinji and Qian, Yanmin},
  year          = {2026},
  eprint        = {2601.18438},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2601.18438}
}
```

---

## License

[MIT](LICENSE).
