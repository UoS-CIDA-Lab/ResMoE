"""Official G-MoEfication input-weight grouping, shared by baseline experiments."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

GROUPING_REFERENCE = (
    "thnkinbtfly/G-MoEfication@5c7af5de5e9d094e9070f907ee3431c2202a5f36:"
    "moefication/utils.py:ParamSplit.split"
)


def gmoe_weight_groups(
    weight: torch.Tensor, groups: int, dev: str, seed: int = 0,
    cache_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize rows and solve equal-size constrained k-means, as upstream.

    A cache is valid only for the exact weight values, seed, expert count,
    implementation and clustering package. Model weights are never modified.
    """
    from importlib.metadata import version
    from k_means_constrained import KMeansConstrained
    from sklearn.preprocessing import normalize

    if weight.ndim != 2 or groups <= 0 or weight.shape[0] == 0 or weight.shape[0] % groups:
        raise ValueError("Input projection must have one row per neuron and equal-size experts")
    if seed < 0:
        raise ValueError("Grouping seed must be nonnegative")
    rows = weight.detach().float().cpu().numpy()
    if not np.isfinite(rows).all():
        raise ValueError("Input-weight rows must be finite")
    size = weight.shape[0] // groups
    metadata = dict(
        reference=GROUPING_REFERENCE, groups=groups, seed=seed,
        weights_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        package_version=version("k-means-constrained"),
    )
    if cache_path is not None and cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError(f"Grouping cache does not match this experiment: {cache_path}")
        labels = saved["labels"].numpy()
    else:
        vectors = normalize(rows)
        clustering = KMeansConstrained(
            n_clusters=groups, size_min=size, size_max=size, random_state=seed,
        )
        clustering.fit(vectors, None)
        labels = np.asarray(clustering.labels_, dtype=np.int64)
    if labels.shape != (weight.shape[0],) or np.any(labels < 0) or np.any(labels >= groups):
        raise RuntimeError("Constrained k-means returned invalid group labels")
    if not np.all(np.bincount(labels, minlength=groups) == size):
        raise RuntimeError("Constrained k-means violated equal expert sizes")
    if cache_path is not None and not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        torch.save(dict(metadata=metadata, labels=torch.from_numpy(labels.copy())), temporary)
        temporary.replace(cache_path)
    return torch.full((groups,), float(size), device=dev), torch.from_numpy(labels.copy()).to(dev)
