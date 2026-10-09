"""OPT representative ablation with the selection rule held fixed.
At each keep rate, compare zero, unconditional mean and conditional mean at
both group and neuron granularity. Selection always uses |activation|*||v||;
changing the representative does not change the selection scoring formula.
The conditional mean is fitted on dense calibration activations only.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer, OPTDecoder, OPTModel, OPTForCausalLM

MODEL = os.environ.get("HHMODEL", "facebook/opt-1.3b")
SEED = int(os.environ.get("SEED", "0"))
N_CALIB = int(os.environ.get("NCALIB", "8192"))
N_EVAL = int(os.environ.get("NEVAL", "4096"))
CHUNK = 512
K = 64
KEEPS = [float(v) for v in os.environ.get("KEEPS", "0.50,0.25,0.10").split(",")]
OUT = os.environ.get("OUT", "")


class Granularity(Enum):
    GROUP = "group"
    NEURON = "neuron"


class Representative(Enum):
    ZERO = "zero"
    MEAN = "mean"
    CONDITIONAL = "conditional_mean"


@dataclass
class Stats:
    activations: torch.Tensor
    mean: torch.Tensor
    vn: torch.Tensor
    groups: torch.Tensor
    sizes: torch.Tensor


@dataclass
class State:
    granularity: Granularity
    budget: int
    representative: torch.Tensor


def keep_mask(a: torch.Tensor, s: Stats, granularity: Granularity, budget: int) -> torch.Tensor:
    scores = (a.float() * s.vn).square()
    if granularity is Granularity.NEURON:
        # Break zero-activation ties by neuron index to enforce the stated budget.
        order = scores.argsort(dim=1, descending=True, stable=True)[:, :budget]
        return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, order, True)
    group_scores = torch.zeros((a.shape[0], K), device=a.device).index_add_(1, s.groups, scores)
    order = group_scores.argsort(dim=1, descending=True, stable=True)
    sizes = s.sizes[order]
    take = sizes.cumsum(1) - sizes < budget
    groups = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, order, take)
    return groups[:, s.groups]


class ExperimentalOutput(nn.Linear):
    def __init__(self, config: OPTConfig) -> None:
        super().__init__(config.ffn_dim, config.hidden_size, bias=config.enable_bias)
        self.capture = False
        self.activations: list[torch.Tensor] = []
        self.stats: Stats | None = None
        self.state: State | None = None
        self.kept = 0
        self.total = 0

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if self.capture:
            self.activations.append(a.reshape(-1, a.shape[-1]).half().cpu())
        if self.state is not None:
            if self.stats is None:
                raise RuntimeError("Missing calibration")
            flat = a.reshape(-1, a.shape[-1])
            mask = keep_mask(flat, self.stats, self.state.granularity, self.state.budget)
            self.kept += int(mask.sum().item())
            self.total += mask.numel()
            a = torch.where(mask, flat, self.state.representative.to(flat.dtype)).reshape_as(a)
        return F.linear(a, self.weight, self.bias)


class ExperimentalLayer(OPTDecoderLayer):
    def __init__(self, config: OPTConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.fc2 = ExperimentalOutput(config)


class ExperimentalDecoder(OPTDecoder):
    def __init__(self, config: OPTConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList([ExperimentalLayer(config, i) for i in range(config.num_hidden_layers)])


class ExperimentalModel(OPTModel):
    def __init__(self, config: OPTConfig) -> None:
        super().__init__(config)
        self.decoder = ExperimentalDecoder(config)


class ExperimentalLM(OPTForCausalLM):
    def __init__(self, config: OPTConfig) -> None:
        super().__init__(config)
        self.model = ExperimentalModel(config)


def kmeans(x: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(SEED)
    centers = x[torch.randperm(x.shape[0], generator=generator, device=x.device)[:K]].clone()
    labels = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
    for _ in range(12):
        labels = torch.cdist(x, centers).argmin(1)
        for g in range(K):
            if (labels == g).any():
                centers[g] = x[labels == g].mean(0)
    return labels


def main() -> None:
    from datasets import load_dataset
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt").input_ids[0]
    if ids.numel() < N_CALIB + N_EVAL:
        raise ValueError("Insufficient corpus tokens")
    model = ExperimentalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    outputs: list[ExperimentalOutput] = []
    for layer in model.model.decoder.layers:
        if not isinstance(layer, ExperimentalLayer):
            raise TypeError("Unexpected decoder layer")
        outputs.append(layer.fc2)
    torch.set_grad_enabled(False)
    metadata = dict(model=MODEL, revision=model.config._commit_hash, seed=SEED,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(), groups=K,
                    dataset="Salesforce/wikitext", config="wikitext-2-raw-v1", split="test",
                    fingerprint=ds._fingerprint, calib_tokens=N_CALIB, eval_tokens=N_EVAL,
                    chunk=CHUNK, selection="activation magnitude times output-weight norm, independent of representative",
                    tie_rule="exact top-budget selection, stable ties by neuron index",
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    tokens_sha256=hashlib.sha256(ids[:N_CALIB + N_EVAL].numpy().tobytes()).hexdigest())
    print(json.dumps(metadata), flush=True)
    rows: list[dict[str, object]] = []

    def evaluate(tag: str) -> dict[str, object]:
        for output in outputs:
            output.kept = output.total = 0
        loss = 0.0
        tokens = 0
        first_loss = 0.0
        first_tokens = 0
        for off in range(0, N_EVAL, CHUNK):
            seq = ids[N_CALIB + off:N_CALIB + min(off + CHUNK, N_EVAL)].unsqueeze(0).to(dev)
            value = F.cross_entropy(model(seq).logits[0, :-1].float(), seq[0, 1:], reduction="sum").item()
            loss += value
            tokens += seq.shape[1] - 1
            if off < 1024:
                first_loss += value
                first_tokens += seq.shape[1] - 1
        total = sum(o.total for o in outputs)
        result: dict[str, object] = dict(tag=tag, ppl=math.exp(loss / tokens),
                                         ppl_first1024=math.exp(first_loss / first_tokens),
                                         actual_keep=sum(o.kept for o in outputs) / total if total else 1.0)
        print(json.dumps(result), flush=True)
        return result

    dense = evaluate("dense")
    for output in outputs:
        output.capture = True
    for off in range(0, N_CALIB, CHUNK):
        model(ids[off:off + CHUNK].unsqueeze(0).to(dev))
    for li, layer in enumerate(model.model.decoder.layers):
        if not isinstance(layer, ExperimentalLayer):
            raise TypeError("Unexpected decoder layer")
        output = layer.fc2
        output.capture = False
        a = torch.cat(output.activations)
        output.activations.clear()
        groups = kmeans(F.normalize(layer.fc1.weight.detach().float(), dim=1))
        sizes = torch.bincount(groups, minlength=K)
        output.stats = Stats(a, a.float().to(dev).mean(0), output.weight.detach().float().norm(dim=0), groups, sizes)
        print(f"Prepared layer {li + 1}/{len(outputs)}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    def save() -> None:
        if OUT:
            Path(OUT).write_text(json.dumps(dict(metadata=metadata, dense=dense, rows=rows), indent=2))

    save()
    for keep in KEEPS:
        budget = max(1, round(keep * model.config.ffn_dim))
        for granularity in Granularity:
            conditional: list[torch.Tensor] = []
            for output in outputs:
                s = output.stats
                if s is None:
                    raise RuntimeError("Missing calibrated state")
                a = s.activations.float().to(dev)
                dropped = ~keep_mask(a, s, granularity, budget)
                rep = (a * dropped).sum(0) / dropped.sum(0).clamp(min=1)
                conditional.append(rep)
                del a, dropped
            for representative in Representative:
                for li, output in enumerate(outputs):
                    s = output.stats
                    if s is None:
                        raise RuntimeError("Missing calibrated state")
                    if representative is Representative.ZERO:
                        rep = torch.zeros_like(s.mean)
                    elif representative is Representative.MEAN:
                        rep = s.mean
                    else:
                        rep = conditional[li]
                    output.state = State(granularity, budget, rep)
                tag = f"keep={keep} granularity={granularity.value} representative={representative.value}"
                row = evaluate(tag)
                row.update(keep=keep, granularity=granularity.value, representative=representative.value)
                rows.append(row)
                save()
    print("Completed controlled representative comparison.", flush=True)


if __name__ == "__main__":
    main()
