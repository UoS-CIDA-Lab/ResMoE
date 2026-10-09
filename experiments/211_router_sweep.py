"""Controlled shared-expert and adaptive-budget comparisons on Qwen2 models.
Groups, representatives and fitted routers are reused across shared fractions and
allocation rules. Fixed/global allocation executes the same number of groups on
average; a causal threshold is calibrated on router data and reports actual keep.
METRIC=ppl uses MBPP; METRIC=superglue uses WikiText-103 and six zero-shot tasks.
No output correction is applied. Run only on GPUs explicitly selected by the caller.
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
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,
)

MODEL = os.environ.get("HHMODEL", "Qwen/Qwen2.5-Coder-1.5B")
SEED = int(os.environ.get("SEED", "0"))
N_STRUCT = int(os.environ.get("NSTRUCT", "8192"))
SWEEP = [int(v) for v in os.environ.get("SWEEP", "8192").split(",")]
N_EVAL = int(os.environ.get("NEVAL", "4096"))
CHUNK = 512
K = 64
RFEAT = 512
STEPS = int(os.environ.get("STEPS", "3000"))
KEEPS = [float(v) for v in os.environ.get("KEEPS", "0.85,0.50,0.25").split(",")]
SHARED = [float(v) for v in os.environ.get("SHARED", "0,0.6").split(",")]
OUT = os.environ.get("OUT", "")


class Metric(Enum):
    PPL = "ppl"
    SUPERGLUE = "superglue"


class Selector(Enum):
    ORACLE = "oracle"
    ROUTER = "router"


class Allocation(Enum):
    FIXED = "fixed"
    GLOBAL = "global"
    THRESHOLD = "threshold"


METRIC = Metric(os.environ.get("METRIC", "ppl"))
ALLOCATIONS = [Allocation(v) for v in os.environ.get("BUDGETS", "fixed,global,threshold").split(",")]


@dataclass
class Stats:
    mean: torch.Tensor
    vn: torch.Tensor
    groups: torch.Tensor
    xbar: torch.Tensor
    projection: torch.Tensor
    features: torch.Tensor
    targets: torch.Tensor
    oracle_scores: torch.Tensor


@dataclass
class State:
    selector: Selector
    allocation: Allocation
    count: int
    shared: torch.Tensor
    threshold: float


def group_selection(scores: torch.Tensor, state: State) -> torch.Tensor:
    n, k = scores.shape
    selected = torch.zeros((n, k), dtype=torch.bool, device=scores.device)
    selected[:, state.shared] = True
    pool = torch.ones(k, dtype=torch.bool, device=scores.device)
    pool[state.shared] = False
    candidates = pool.nonzero().flatten()
    remaining = state.count - state.shared.numel()
    if remaining == 0:
        return selected
    routed = scores[:, candidates]
    if state.allocation is Allocation.FIXED:
        chosen = routed.argsort(dim=1, descending=True, stable=True)[:, :remaining]
        selected.scatter_(1, candidates[chosen], True)
    elif state.allocation is Allocation.GLOBAL:
        order = routed.reshape(-1).argsort(descending=True, stable=True)[:n * remaining]
        rows = torch.div(order, candidates.numel(), rounding_mode="floor")
        selected[rows, candidates[order % candidates.numel()]] = True
    else:
        selected[:, candidates] = routed >= state.threshold
    return selected


class ExperimentalMLP(Qwen2MLP):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config)
        self.router = nn.Sequential(nn.Linear(RFEAT, 512), nn.GELU(), nn.Linear(512, K)).float()
        self.stats: Stats | None = None
        self.state: State | None = None
        self.capture = False
        self.inputs: list[torch.Tensor] = []
        self.activations: list[torch.Tensor] = []
        self.kept = 0
        self.total = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if self.capture:
            self.inputs.append(x.reshape(-1, x.shape[-1]).half().cpu())
            self.activations.append(a.reshape(-1, a.shape[-1]).half().cpu())
        if self.state is None:
            return self.down_proj(a)
        s = self.stats
        if s is None:
            raise RuntimeError("Selection requested before calibration")
        af = a.reshape(-1, a.shape[-1])
        if self.state.selector is Selector.ORACLE:
            per = ((af.float() - s.mean) * s.vn).square()
            scores = torch.zeros((af.shape[0], K), device=x.device).index_add_(1, s.groups, per)
        else:
            z = (x.reshape(-1, x.shape[-1]).float() - s.xbar) @ s.projection
            scores = self.router(z)
        selected = group_selection(scores, self.state)
        mask = selected[:, s.groups]
        self.kept += int(mask.sum().item())
        self.total += mask.numel()
        masked = torch.where(mask, af, s.mean.to(af.dtype))
        return self.down_proj(masked.reshape_as(a))


# These subclasses establish all execution modules during construction; the
# pretrained parameter names remain compatible with the original checkpoint.
class ExperimentalLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.mlp = ExperimentalMLP(config)


class ExperimentalModel(Qwen2Model):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList([ExperimentalLayer(config, i) for i in range(config.num_hidden_layers)])


class ExperimentalLM(Qwen2ForCausalLM):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__(config)
        self.model = ExperimentalModel(config)


def balanced_groups(weights: torch.Tensor) -> torch.Tensor:
    x = F.normalize(weights.float(), dim=1)
    gen = torch.Generator(device=x.device).manual_seed(SEED)
    centers = x[torch.randperm(x.shape[0], generator=gen, device=x.device)[:K]].clone()
    for _ in range(12):
        labels = torch.cdist(x, centers).argmin(1)
        for g in range(K):
            if (labels == g).any():
                centers[g] = x[labels == g].mean(0)
    distances = torch.cdist(x, centers)
    preferences = distances.argsort(dim=1, stable=True)
    first = distances.gather(1, preferences[:, :2])
    order = (first[:, 1] - first[:, 0]).argsort(descending=True, stable=True).tolist()
    capacity = x.shape[0] // K
    if x.shape[0] % K:
        raise ValueError("Exact matched group budgets require FFN width divisible by K")
    counts = [0] * K
    assigned = [0] * x.shape[0]
    for i in order:
        for g in preferences[i].tolist():
            if counts[g] < capacity:
                assigned[i] = g
                counts[g] += 1
                break
    return torch.tensor(assigned, device=x.device)


def fit_router(mlp: ExperimentalMLP, n: int, layer: int) -> None:
    s = mlp.stats
    if s is None:
        raise RuntimeError("Missing router data")
    torch.manual_seed(SEED + layer)
    for module in mlp.router:
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    device = s.mean.device
    x = s.features[:n].float().to(device)
    y = s.targets[:n].float().to(device)
    opt = torch.optim.Adam(mlp.router.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(SEED)
    with torch.enable_grad():
        for _ in range(STEPS):
            indices = torch.randint(0, n, (2048,), generator=generator, device=device)
            opt.zero_grad()
            F.mse_loss(mlp.router(x[indices]), y[indices]).backward()
            opt.step()
    mlp.router.eval()


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
    corpus_fingerprints: dict[str, str] = {}
    if METRIC is Metric.PPL:
        code: list[str] = []
        for split in ["train", "test", "validation", "prompt"]:
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
            corpus_fingerprints[split] = ds._fingerprint
            code.extend(ds["code"])
        ids = tok("\n\n".join(code), return_tensors="pt").input_ids[0]
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        corpus_fingerprints["train"] = ds._fingerprint
        pieces: list[torch.Tensor] = []
        count = 0
        for text in ds["text"]:
            if not text.strip():
                continue
            piece = tok(text, return_tensors="pt").input_ids[0]
            pieces.append(piece)
            count += piece.numel()
            if count >= N_STRUCT + max(SWEEP) + N_EVAL:
                break
        ids = torch.cat(pieces)
    needed = N_STRUCT + max(SWEEP) + N_EVAL
    if ids.numel() < needed:
        raise ValueError(f"Corpus too short: {ids.numel()} < {needed}")
    model = ExperimentalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    model.config.use_cache = False
    mlps: list[ExperimentalMLP] = []
    for layer in model.model.layers:
        if not isinstance(layer, ExperimentalLayer):
            raise TypeError("Unexpected decoder layer")
        mlps.append(layer.mlp)
        layer.mlp.router.float()
    torch.set_grad_enabled(False)
    metadata = dict(model=MODEL, revision=model.config._commit_hash, seed=SEED,
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    corpus_fingerprints=corpus_fingerprints,
                    tokens_sha256=hashlib.sha256(ids[:needed].numpy().tobytes()).hexdigest(),
                    n_struct=N_STRUCT, router_tokens=SWEEP, eval_tokens=N_EVAL,
                    eval_start=N_STRUCT + max(SWEEP), groups=K, steps=STEPS,
                    correction=False, metric=METRIC.value, shared_fractions=SHARED,
                    allocations=[a.value for a in ALLOCATIONS], keeps=KEEPS)
    print(json.dumps(metadata), flush=True)
    rows: list[dict[str, object]] = []

    def evaluate(tag: str) -> dict[str, object]:
        for mlp in mlps:
            mlp.kept = mlp.total = 0
        if METRIC is Metric.PPL:
            total_loss = 0.0
            total_tokens = 0
            first_loss = 0.0
            first_tokens = 0
            start = N_STRUCT + max(SWEEP)
            for off in range(0, N_EVAL, CHUNK):
                seq = ids[start + off:start + min(off + CHUNK, N_EVAL)].unsqueeze(0).to(dev)
                loss = F.cross_entropy(model(seq).logits[0, :-1].float(), seq[0, 1:], reduction="sum").item()
                total_loss += loss
                total_tokens += seq.shape[1] - 1
                if off < 1024:
                    first_loss += loss
                    first_tokens += seq.shape[1] - 1
            values: dict[str, object] = dict(ppl=math.exp(total_loss / total_tokens),
                                            ppl_first1024=math.exp(first_loss / first_tokens))
        else:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            tasks = ["boolq", "cb", "copa", "rte", "wic", "wsc"]
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=4)
            result = lm_eval.simple_evaluate(model=lm, tasks=tasks, num_fewshot=0,
                                            random_seed=SEED, numpy_random_seed=SEED,
                                            torch_random_seed=SEED, fewshot_random_seed=SEED,
                                            verbosity="ERROR")
            scores = {t: result["results"][t]["acc,none"] for t in tasks}
            values = dict(superglue=sum(scores.values()) / len(scores), tasks=scores)
            del lm, result
        total = sum(m.total for m in mlps)
        values.update(tag=tag, actual_keep=sum(m.kept for m in mlps) / total if total else 1.0)
        print(json.dumps(values), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        return values

    dense = evaluate("dense")
    for mlp in mlps:
        mlp.capture = True
    for off in range(0, N_STRUCT + max(SWEEP), CHUNK):
        model(ids[off:off + CHUNK].unsqueeze(0).to(dev))
    for li, mlp in enumerate(mlps):
        mlp.capture = False
        x = torch.cat(mlp.inputs).float().to(dev)
        a = torch.cat(mlp.activations).float().to(dev)
        mlp.inputs.clear()
        mlp.activations.clear()
        mean = a[:N_STRUCT].mean(0)
        xbar = x[:N_STRUCT].mean(0)
        _, _, vt = torch.linalg.svd(x[:N_STRUCT] - xbar, full_matrices=False)
        projection = vt[:RFEAT].T.contiguous()
        groups = balanced_groups(mlp.gate_proj.weight.detach())
        wd = mlp.down_proj.weight.detach().float()
        vn = wd.norm(dim=0)
        delta = a[N_STRUCT:] - mean
        oracle = torch.zeros((delta.shape[0], K), device=dev).index_add_(1, groups, (delta * vn).square())
        targets = torch.stack([(delta[:, groups == g] @ wd[:, groups == g].T).norm(dim=1)
                               for g in range(K)], dim=1)
        features = (x[N_STRUCT:] - xbar) @ projection
        mlp.stats = Stats(mean, vn, groups, xbar, projection, features.half().cpu(),
                          targets.half().cpu(), oracle.half().cpu())
        del x, a, delta, targets, features, oracle, wd, vt
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Prepared layer {li + 1}/{len(mlps)}", flush=True)

    def save() -> None:
        if OUT:
            Path(OUT).write_text(json.dumps(dict(metadata=metadata, dense=dense, rows=rows), indent=2))

    save()
    for n in SWEEP:
        for li, mlp in enumerate(mlps):
            fit_router(mlp, n, li)
            print(f"Fitted router {li + 1}/{len(mlps)} on {n} tokens", flush=True)
        for keep in KEEPS:
            count = max(1, min(K, round(keep * K)))
            for shared_fraction in SHARED:
                for selector in Selector:
                    for allocation in ALLOCATIONS:
                        for mlp in mlps:
                            s = mlp.stats
                            if s is None:
                                raise RuntimeError("Missing calibrated state")
                            frequency = torch.zeros(K, device=dev)
                            top = s.oracle_scores[:n].float().to(dev).argsort(dim=1, descending=True, stable=True)[:, :count]
                            frequency.scatter_add_(0, top.reshape(-1), torch.ones(top.numel(), device=dev))
                            shared = frequency.argsort(descending=True, stable=True)[:round(shared_fraction * count)]
                            remaining_pool = torch.ones(K, dtype=torch.bool, device=dev)
                            remaining_pool[shared] = False
                            if selector is Selector.ORACLE:
                                calibration = s.oracle_scores[:n].float().to(dev)
                            else:
                                calibration = mlp.router(s.features[:n].float().to(dev))
                            required = n * (count - shared.numel())
                            ranked = calibration[:, remaining_pool].reshape(-1).sort(dim=0, descending=True, stable=True).values
                            threshold = float(ranked[required - 1]) if required else math.inf
                            mlp.state = State(selector, allocation, count, shared, threshold)
                        tag = f"keep={keep} shared={shared_fraction} selector={selector.value} budget={allocation.value}"
                        row = evaluate(tag)
                        row.update(keep=keep, group_count=count, shared_fraction=shared_fraction,
                                   selector=selector.value, budget=allocation.value, router_tokens=n)
                        rows.append(row)
                        save()
    print("Completed controlled comparisons.", flush=True)


if __name__ == "__main__":
    main()
