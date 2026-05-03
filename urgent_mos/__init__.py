"""URGENT-MOS: speech quality (MOS / CMOS) prediction.

Public API:
    >>> from urgent_mos import infer, infer_pairs, load_model_from_checkpoint
    >>> model = load_model_from_checkpoint("urgent-challenge/urgent-mos-f1c1m5dcorpus", "cuda")
    >>> scores = infer(model, ["audio.wav"])
"""
from importlib.metadata import PackageNotFoundError, version

from urgent_mos.api.infer import infer, infer_pairs
from urgent_mos.utils import load_model_from_checkpoint

try:
    __version__ = version("urgent_mos")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "infer",
    "infer_pairs",
    "load_model_from_checkpoint",
]
