"""Experiment 248 — seeded mBERT grouping/correction decomposition on NER/POS.
G-MoEfication grouping uses normalized FFN input weights and KMeansConstrained,
matching upstream ParamSplit.split (commit 5c7af5de5e9d094e9070f907ee3431c2202a5f36).
Also evaluates activation-pattern grouping, selection-pattern grouping, correction,
and the neuron-oracle reference. All methods use the same retained-neuron budget.
Evaluation masks a dense FFN; the budget describes retained contributions,
not measured sparse execution or speedup.
This is an adapted 9-language, pooled-metric, group-oracle comparison, not a
reproduction of the published 42-language scores or learned selector.
Defaults: FINETUNE_SEED=0, post-training SEED=0, K=64, full
75/50/35/25/20/15% sweep, two fine-tuning epochs. For repeated grouping/correction
runs, hold FINETUNE_SEED and DENSE_CHECKPOINT fixed and vary only SEED.
Run: TASK=ner python3 experiments/248_mbert_modular.py   (and TASK=pos).
RESULT_DIR stores the dense checkpoint, calibration, groupings, corrections and
incremental results.json. DENSE_CHECKPOINT explicitly reuses a compatible saved
fine-tuned checkpoint; no training is performed in that mode.
"""
from __future__ import annotations
import sys, pathlib, gc, os, json, random, math, hashlib
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version
from typing import TypedDict
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from experiments.baseline_grouping import gmoe_weight_groups


class Task(Enum):
    NER = "ner"
    POS = "pos"


class Selection(Enum):
    OFF = "off"
    GROUP = "group"
    NEURON = "neuron"
    CORRECTION = "correction"


class Method(Enum):
    GMOE = "gmoe_weight_grouping"
    COACTIVATION = "coactivation"
    PATTERN = "selection_pattern"
    CORRECTION = "correction"
    NEURON = "neuron_oracle"


class LayerSelection(TypedDict, total=False):
    mode: Selection
    B: int
    vn: torch.Tensor
    abar: torch.Tensor
    gsz: torch.Tensor
    gf: torch.Tensor
    Br: torch.Tensor
    xbar: torch.Tensor
    P: torch.Tensor
    pred: nn.Module

MODEL = "bert-base-multilingual-cased"
TASK = Task(os.environ.get("TASK", "ner"))
FINETUNE_SEED = int(os.environ.get("FINETUNE_SEED", "0"))
SEED = int(os.environ.get("SEED", "0"))
LANGS = os.environ.get("LANGS", "en,de,es,fr,nl,ar,zh,hi,ru").split(",")
UDNAME = {"en": "English", "de": "German", "es": "Spanish", "fr": "French", "nl": "Dutch",
          "ar": "Arabic", "zh": "Chinese", "hi": "Hindi", "ru": "Russian"}
SEQ = 128; EPOCHS = int(os.environ.get("EPOCHS", "2"))
K = int(os.environ.get("K", "64")); RCORR = 128; RFEAT = 512
KEEPS = [float(x) for x in os.environ.get("KEEPS", "0.75,0.50,0.35,0.25,0.20,0.15").split(",")]
RESULT_DIR = pathlib.Path(os.environ.get("RESULT_DIR", f"experiments/results/248_{TASK.value}_seed{SEED}_K{K}"))
DENSE_CHECKPOINT = pathlib.Path(os.environ["DENSE_CHECKPOINT"]) if "DENSE_CHECKPOINT" in os.environ else None
NER_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
POS_LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
              "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def write_json(path: pathlib.Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), indent=2) + "\n")
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def executed_budget(keep: float, neurons: int, groups: int) -> int:
    if not 0 < keep <= 1 or groups <= 0 or neurons % groups:
        raise ValueError("Keep must be in (0,1] and the expert count must divide FFN width")
    requested = int(round(keep * neurons))
    return max(1, math.ceil(requested / (neurons // groups))) * (neurons // groups)


def kmeans_centroids(X, k, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = X[m].mean(0)
    return c


def weighted_kmeans_centroids(X, w, k, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                wj = w[m]; c[j] = (X[m] * wj.unsqueeze(1)).sum(0) / wj.sum().clamp(min=1e-6)
    return c


def balanced_assign(X, centroids):
    D = torch.cdist(X, centroids); n, k = D.shape
    cap = (n + k - 1) // k
    pref = D.argsort(1); d12 = D.gather(1, pref[:, :2]); regret = d12[:, 1] - d12[:, 0]
    order = regret.argsort(descending=True).tolist(); pref_l = pref.tolist()
    counts = [0] * k; assign = [0] * n
    for i in order:
        for c in pref_l[i]:
            if counts[c] < cap:
                assign[i] = c; counts[c] += 1; break
    return torch.tensor(assign, device=X.device, dtype=torch.long)


def mlp_fit(X: torch.Tensor, Y: torch.Tensor, dev: str, steps: int = 3000,
            hidden: int = 512, lr: float = 3e-3, bs: int = 2048, seed: int = 0) -> nn.Sequential:
    devices = [torch.cuda.current_device()] if dev == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])).to(dev).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.enable_grad():
        for _ in range(steps):
            bi = torch.randint(0, X.shape[0], (bs,), generator=gen, device=dev)
            opt.zero_grad(); F.mse_loss(net(X[bi]), Y[bi]).backward(); opt.step()
    return net.eval()


def keep_topB_neuron(score: torch.Tensor, B: int) -> torch.Tensor:
    if B < 1 or B > score.shape[1]:
        raise ValueError("Neuron budget must be between one and FFN width")
    # Thresholding can exceed the budget when scores tie.
    indices = score.topk(B, dim=1).indices
    return torch.zeros_like(score, dtype=torch.bool).scatter_(1, indices, True)


def keep_topB_group(score_g, gsz, gf, B):
    N, Kc = score_g.shape
    order = score_g.argsort(1, descending=True); so = gsz[order]
    keep_ord = (so.cumsum(1) - so) < B
    selg = torch.zeros(N, Kc, dtype=torch.bool, device=score_g.device).scatter_(1, order, keep_ord)
    return selg[:, gf]


def oracle_mask(a, abar, vn, gsz, gf, B):
    per = ((a - abar) * vn) ** 2
    sg = torch.zeros(a.shape[0], gsz.shape[0], device=a.device).index_add_(1, gf, per)
    return keep_topB_group(sg, gsz, gf, B).to(a.dtype)


def gsizes(gl, dev):
    g = torch.zeros(K, device=dev)
    for j in range(K):
        g[j] = (gl == j).sum()
    return g


def main() -> None:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from datasets import load_dataset, concatenate_datasets
    from seqeval.metrics import f1_score
    seed_everything(FINETUNE_SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if (RESULT_DIR / "results.json").exists():
        raise FileExistsError(f"Choose a new RESULT_DIR to preserve existing results: {RESULT_DIR}")
    if EPOCHS < 1 or not KEEPS or len(set(KEEPS)) != len(KEEPS):
        raise ValueError("Require positive epochs and a nonempty unique keep sweep")
    for keep in KEEPS:
        executed_budget(keep, 3072, K)
    source_snapshot = pathlib.Path(__file__).read_bytes()
    (RESULT_DIR / "source_snapshot.py").write_bytes(source_snapshot)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    tagfield = "ner_tags" if TASK is Task.NER else "pos_tags"

    def load_split(lang, split):
        if TASK is Task.NER:
            return load_dataset("unimelb-nlp/wikiann", lang, split=split)
        return load_dataset("google/xtreme", f"udpos.{UDNAME[lang]}", split=split)
    tr = concatenate_datasets([load_split(l, "train") for l in LANGS]).shuffle(seed=FINETUNE_SEED)
    te = concatenate_datasets([load_split(l, "test") for l in LANGS])
    # derive label names from the dataset (robust to schema/order)
    feat = tr.features[tagfield]
    names = getattr(getattr(feat, "feature", feat), "names", None)
    LABELS = list(names) if names else (NER_LABELS if TASK is Task.NER else POS_LABELS)
    id2lab = {i: l for i, l in enumerate(LABELS)}
    print(f"  TASK={TASK.value} finetune_seed={FINETUNE_SEED} post_training_seed={SEED} K={K} langs={LANGS} train={len(tr)} test={len(te)}", flush=True)
    training_config = dict(model=MODEL, task=TASK.value, languages=LANGS, seed=FINETUNE_SEED,
                           epochs=EPOCHS, sequence_length=SEQ, labels=LABELS, train_rows=len(tr))
    metadata = dict(training_config=training_config, post_training_seed=SEED, expert_count=K, keeps=KEEPS,
                    source_sha256=hashlib.sha256(source_snapshot).hexdigest(),
                    versions={p: version(p) for p in ("torch", "transformers", "datasets", "numpy", "scikit-learn", "k-means-constrained", "ortools")},
                    grouping_reference="thnkinbtfly/G-MoEfication@5c7af5de5e9d094e9070f907ee3431c2202a5f36:moefication/utils.py:ParamSplit.split",
                    protocol=f"{len(LANGS)} configured languages, pooled metrics, oracle selection, mean representative; published scores use 42 languages and a learned selector",
                    padding_in_calibration=True, calibration_examples=4000, max_calibration_tokens_per_layer=16384,
                    seed_note="Fine-tuning seed is independent of calibration sampling, clustering initialization and correction seeds; GPU reductions do not guarantee bitwise determinism across devices/versions",
                    dense_checkpoint_source=str(DENSE_CHECKPOINT) if DENSE_CHECKPOINT else MODEL,
                    started_at=datetime.now(timezone.utc).isoformat())
    write_json(RESULT_DIR / "metadata.json", metadata)

    def encode(batch):
        toks = batch["tokens"]
        out = tok(toks, is_split_into_words=True, truncation=True, max_length=SEQ, padding="max_length")
        labels = []
        for i, tags in enumerate(batch[tagfield]):
            wids = out.word_ids(i); prev = None; lab = []
            for w in wids:
                if w is None or w == prev:
                    lab.append(-100)
                else:
                    lab.append(int(tags[w]))
                prev = w
            labels.append(lab)
        out["labels"] = labels
        return out
    tr_enc = tr.map(encode, batched=True, remove_columns=tr.column_names)
    te_enc = te.map(encode, batched=True, remove_columns=te.column_names)
    tr_enc.set_format("torch"); te_enc.set_format("torch")
    if DENSE_CHECKPOINT is not None:
        saved = json.loads((DENSE_CHECKPOINT / "training_config.json").read_text())
        if saved != training_config:
            raise ValueError("Dense checkpoint fine-tuning configuration does not match this run")
    model = AutoModelForTokenClassification.from_pretrained(
        str(DENSE_CHECKPOINT) if DENSE_CHECKPOINT else MODEL, num_labels=len(LABELS)).to(dev)

    from torch.utils.data import DataLoader
    if DENSE_CHECKPOINT is None:
        dl = DataLoader(tr_enc, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(FINETUNE_SEED))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        model.train()
        for ep in range(EPOCHS):
            tot = 0.0; nb = 0
            for b in dl:
                o = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev),
                          labels=b["labels"].to(dev))
                opt.zero_grad(); o.loss.backward(); opt.step(); tot += o.loss.item(); nb += 1
                if nb % 1000 == 0:
                    print(f"  epoch {ep} step {nb}/{len(dl)} loss {tot/nb:.4f}", flush=True)
            print(f"  epoch {ep} loss {tot/nb:.4f}", flush=True)
        checkpoint = RESULT_DIR / "dense_checkpoint"
        model.save_pretrained(checkpoint); tok.save_pretrained(checkpoint)
        write_json(checkpoint / "training_config.json", training_config)
    checkpoint_source = DENSE_CHECKPOINT if DENSE_CHECKPOINT is not None else RESULT_DIR / "dense_checkpoint"
    metadata['dense_checkpoint_sha256'] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(checkpoint_source.glob("*.safetensors"))}
    write_json(RESULT_DIR / "metadata.json", metadata)
    # Calibration sampling is independent of whether training was reused.
    seed_everything(SEED)
    model.eval(); torch.set_grad_enabled(False)

    enc = model.base_model.encoder.layer; nL = len(enc)
    inproj = [enc[li].intermediate.dense for li in range(nL)]
    outproj = [enc[li].output.dense for li in range(nL)]
    CFG: dict[int, LayerSelection] = {li: {'mode': Selection.OFF} for li in range(nL)}
    XC: dict[int, torch.Tensor | None] = {li: None for li in range(nL)}

    def gpre(li):
        def h(_m, a):
            XC[li] = a[0].reshape(-1, a[0].shape[-1])
        return h

    def dpre(li):
        def h(_m, args):
            cfg = CFG[li]
            if cfg['mode'] is Selection.OFF:
                return None
            a = args[0]; sh = a.shape; af = a.reshape(-1, sh[-1])
            if cfg['mode'] is Selection.NEURON:
                m = keep_topB_neuron((af.float() - cfg['abar']).abs() * cfg['vn'], cfg['B']).to(a.dtype)
            else:
                m = oracle_mask(af.float(), cfg['abar'], cfg['vn'], cfg['gsz'], cfg['gf'], cfg['B']).to(a.dtype)
            return ((af * m + cfg['abar'].to(a.dtype) * (1 - m)).reshape(sh),) + args[1:]
        return h

    def dpost(li):
        def h(_m, args, output):
            cfg = CFG[li]
            if cfg['mode'] is not Selection.CORRECTION:
                return None
            sh = output.shape
            x = XC[li]
            if x is None:
                raise RuntimeError("Correction requires the observed FFN input")
            z = (x.float() - cfg['xbar']) @ cfg['P']
            return (output.reshape(-1, sh[-1]) + (cfg['pred'](z) @ cfg['Br'].T).to(output.dtype)).reshape(sh)
        return h
    for li in range(nL):
        inproj[li].register_forward_pre_hook(gpre(li))
        outproj[li].register_forward_pre_hook(dpre(li))
        outproj[li].register_forward_hook(dpost(li))

    from torch.utils.data import DataLoader as DL
    te_dl = DL(te_enc, batch_size=64); cal_dl = DL(tr_enc.select(range(min(4000, len(tr_enc)))), batch_size=64)

    def metric_eval():
        preds, refs = [], []
        for b in te_dl:
            ii = b["input_ids"].to(dev); am = b["attention_mask"].to(dev); lb = b["labels"]
            pr = model(input_ids=ii, attention_mask=am).logits.argmax(-1).cpu()
            for i in range(ii.shape[0]):
                p, r = [], []
                for j in range(ii.shape[1]):
                    if lb[i, j].item() != -100:
                        p.append(id2lab[pr[i, j].item()]); r.append(id2lab[lb[i, j].item()])
                preds.append(p); refs.append(r)
        if TASK is Task.NER:
            return f1_score(refs, preds)
        tp = sum(sum(a == b for a, b in zip(p, r)) for p, r in zip(preds, refs))
        tot = sum(len(r) for r in refs)
        return tp / tot

    capa: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}
    capx: dict[int, list[torch.Tensor]] = {li: [] for li in range(nL)}
    hs = []
    for li in range(nL):
        hs.append(outproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capa[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
        hs.append(inproj[li].register_forward_pre_hook(
            (lambda li: (lambda _m, a: capx[li].append(a[0].reshape(-1, a[0].shape[-1]).half().cpu())))(li)))
    for b in cal_dl:
        model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev))
    for h in hs:
        h.remove()
    dff = capa[0][0].shape[1]
    STR = {}
    for li in range(nL):
        a = torch.cat(capa[li]); x = torch.cat(capx[li])
        if a.shape[0] > 16384:
            idx = torch.randperm(a.shape[0])[:16384]; a = a[idx]; x = x[idx]
        a = a.float().to(dev); x = x.float().to(dev)
        Wd = outproj[li].weight.detach().float().to(dev); Wd = Wd if Wd.shape[1] == dff else Wd.T
        vn = Wd.norm(dim=0); abar = a.mean(0)
        xbar = x.mean(0); _, _, VtX = torch.linalg.svd(x - xbar, full_matrices=False)
        STR[li] = dict(a=a.half().cpu(), x=x.half().cpu(), abar=abar, vn=vn, Wd=Wd.cpu(), xbar=xbar, P=VtX[:RFEAT].T)
        capa[li] = []; capx[li] = []; del a, x, Wd; gc.collect(); torch.cuda.empty_cache()
    print(f"  dff={dff} nL={nL}; harvested", flush=True)
    torch.save(STR, RESULT_DIR / "calibration.pt")

    def grouping_coact(li: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Separate activation-pattern baseline, not upstream weight grouping.
        s = STR[li]; a = s['a'].float().to(dev)
        pat = F.normalize((a - s['abar']).T.contiguous(), dim=1)   # [m, N] activation pattern per neuron
        gl = balanced_assign(pat, kmeans_centroids(pat, K, seed=SEED))
        del a, pat; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def grouping_keep(li: int, B: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = STR[li]; a = s['a'].float().to(dev)
        Bind = keep_topB_neuron((a - s['abar']).abs() * s['vn'], B).float()
        w = ((a - s['abar']).abs() * s['vn']).mean(0)
        gl = balanced_assign(Bind.T.contiguous(), weighted_kmeans_centroids(Bind.T.contiguous(), w, K, seed=SEED))
        del a, Bind; gc.collect(); torch.cuda.empty_cache()
        return gsizes(gl, dev), gl

    def basis_and_pred(li: int, B: int, gsz: torch.Tensor, gf: torch.Tensor) -> tuple[torch.Tensor, nn.Sequential]:
        s = STR[li]; a = s['a'].float().to(dev); abar = s['abar']; Wd = s['Wd'].to(dev)
        m = oracle_mask(a, abar, s['vn'], gsz, gf, B); drop = a - (a * m + abar * (1 - m))
        E = drop @ Wd.T; del a, m, drop, Wd; gc.collect(); torch.cuda.empty_cache()
        Ec = E.cpu(); _, _, Vt = torch.linalg.svd(Ec, full_matrices=False); Br = Vt[:RCORR].T.to(dev)
        z = (s['x'].float().to(dev) - s['xbar']) @ s['P']; pred = mlp_fit(z, E @ Br, dev, seed=SEED + 1000*li + B)
        del E, Ec, Vt, z; gc.collect(); torch.cuda.empty_cache()
        return Br, pred

    def setcfg(GRP: Mapping[int, tuple[torch.Tensor, torch.Tensor]], B: int, mode: Selection,
               BR: Mapping[int, torch.Tensor] | None = None, PRED: Mapping[int, nn.Module] | None = None) -> None:
        if mode is Selection.CORRECTION and (BR is None or PRED is None):
            raise ValueError("Correction requires both its basis and predictor")
        for li in range(nL):
            s = STR[li]; gsz, gf = GRP[li]
            CFG[li].update({'mode': mode, 'B': B, 'vn': s['vn'], 'abar': s['abar'],
                           'gsz': gsz, 'gf': gf, 'xbar': s['xbar'], 'P': s['P']})
            if BR is not None:
                CFG[li]['Br'] = BR[li]
            if PRED is not None:
                CFG[li]['pred'] = PRED[li]

    def off() -> None:
        for li in range(nL):
            CFG[li]['mode'] = Selection.OFF

    unit = "F1" if TASK is Task.NER else "acc"
    dense = metric_eval(); print(f"\n  dense {TASK.value} {unit} = {dense:.4f}\n", flush=True)
    rows: list[dict[str, float | int]] = []
    results: dict[str, object] = dict(status="running", task=TASK.value, metric=unit, dense=dense, measurements=rows)
    write_json(RESULT_DIR / "results.json", results)
    GW = {}
    for li in range(nL):
        print(f"  G-MoEfication weight grouping layer {li+1}/{nL}", flush=True)
        GW[li] = gmoe_weight_groups(inproj[li].weight, K, dev, seed=SEED)
    GC = {li: grouping_coact(li) for li in range(nL)}
    torch.save({'gmoe_weight_grouping': GW, 'coactivation': GC}, RESULT_DIR / "baseline_groups.pt")

    def record(row: dict[str, float | int], method: Method, score: float) -> None:
        row[method.value] = score
        write_json(RESULT_DIR / "results.json", results)
        print(f"  FFN{int(round(row['keep']*100))} {method.value}: {score:.6f}", flush=True)

    for bf in KEEPS:
        B = executed_budget(bf, dff, K)
        row: dict[str, float | int] = dict(keep=bf, executed_neurons=B, actual_keep=B/dff)
        rows.append(row)
        print(f"  keep {bf:.0%}: all methods retain {B}/{dff} neurons ({B/dff:.6%})", flush=True)
        GP = {li: grouping_keep(li, B) for li in range(nL)}
        torch.save(GP, RESULT_DIR / f"selection_groups_keep{int(round(bf*100))}.pt")
        setcfg(GW, B, Selection.GROUP); weight_score = metric_eval(); off(); record(row, Method.GMOE, weight_score)
        setcfg(GC, B, Selection.GROUP); coact = metric_eval(); off(); record(row, Method.COACTIVATION, coact)
        setcfg(GP, B, Selection.GROUP); kp = metric_eval(); off(); record(row, Method.PATTERN, kp)
        setcfg(GP, B, Selection.NEURON); nu = metric_eval(); off(); record(row, Method.NEURON, nu)
        BR = {}; PRED = {}
        for li in range(nL):
            print(f"  keep {bf:.0%} correction layer {li+1}/{nL}", flush=True)
            BR[li], PRED[li] = basis_and_pred(li, B, *GP[li])
        torch.save({'basis': BR, 'predictor_state': {li: pred.state_dict() for li, pred in PRED.items()}},
                   RESULT_DIR / f"correction_keep{int(round(bf*100))}.pt")
        setcfg(GP, B, Selection.CORRECTION, BR, PRED); ours = metric_eval(); off(); record(row, Method.CORRECTION, ours)
        print(f"  FFN{int(round(bf*100))} [{TASK.value} {unit}]: dense {dense:.4f} | G-MoE weight {weight_score:.4f} | "
              f"co-activation {coact:.4f} | selection-pattern {kp:.4f} | +corr {ours:.4f} | neuron {nu:.4f}", flush=True)
    results['status'] = "complete"
    write_json(RESULT_DIR / "results.json", results)
    metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(RESULT_DIR / "metadata.json", metadata)
    print("Grouping gain: G-MoEfication weight grouping -> selection patterns. Correction gain: selection patterns -> +correction.", flush=True)


if __name__ == "__main__":
    main()
