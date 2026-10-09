"""Seed control and provenance for repeated conversion ablations."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


def load_mbpp_code() -> list[str]:
    """Load every paper-protocol split; propagate failures instead of changing data."""
    from datasets import load_dataset

    code: list[str] = []
    for split in ("train", "test", "validation", "prompt"):
        dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
        code.extend(dataset["code"])
    return code


def configure_determinism(seed: int) -> None:
    if seed < 0:
        raise ValueError("SEED must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def report_provenance(
    *, source: str, model: str, revision: str | None, seed: int,
    tokens: torch.Tensor, fingerprints: Mapping[str, str],
    calibration_items: int, evaluation_items: int, chunk: int,
) -> None:
    """Print exact corpus/software provenance before collecting measurements."""
    print(json.dumps(dict(
        record="provenance", model=model, revision=revision, seed=seed,
        python=platform.python_version(), torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        datasets=importlib.metadata.version("datasets"),
        gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        physical_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        reproducibility_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tokens_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),
        corpus_fingerprints=dict(fingerprints), calibration_items=calibration_items,
        evaluation_items=evaluation_items, chunk=chunk,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )), flush=True)


def report_metrics(values: Mapping[str, float | int | str]) -> None:
    """Retain unrounded measurements in execution logs."""
    print(json.dumps(dict(record="measurement", **values)), flush=True)
