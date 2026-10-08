"""Experiment 256 — fine-tune T5-large on a downstream task (SST-2 / MNLI / RACE) in the
text-to-text format, reproducing the ORIGINAL MoEfication paper's setup (the paper fine-tunes
T5: Adam, lr 1e-6, batch 64, 3 epochs; tab:model-performance numbers are from fine-tuned models;
the released demo code skips FT but the paper does not). Saves a checkpoint for 257 to MoEfy.
Grounding target (T5-Large dense acc): SST-2 96.2 / MNLI 89.5 / RACE 81.3.
Non-gated T5 v1.0 (ReLU). Run: TASK=sst2 conda run -n resmoe python3 experiments/256_t5_glue_ft.py
"""
from __future__ import annotations
import os, pathlib, random
import torch

MODEL = os.environ.get("HHMODEL", "google-t5/t5-large")
TASK = os.environ.get("TASK", "sst2")                       # sst2 | mnli | race
OUT = os.environ.get("OUT", str(pathlib.Path(__file__).resolve().parents[1] / "ckpts" / f"t5-large-{TASK}"))
OPT = os.environ.get("OPT", "adafactor")                    # adafactor (T5 standard) | adam
LR = float(os.environ.get("LR", "1e-3"))                    # escalated: paper's 1e-6 Adam underfits MNLI/RACE badly
EPOCHS = int(os.environ.get("EPOCHS", "3"))
SEQ = int(os.environ.get("SEQ", "512" if TASK == "race" else "128"))
BS = int(os.environ.get("BS", "4" if TASK == "race" else "16"))   # micro-batch
ACCUM = int(os.environ.get("ACCUM", "16" if TASK == "race" else "4"))  # BS*ACCUM = 64
MAXTRAIN = int(os.environ.get("MAXTRAIN", "0"))             # 0 = full; else cap (debug)

SST2_LAB = {0: "negative", 1: "positive"}
MNLI_LAB = {0: "entailment", 1: "neutral", 2: "contradiction"}


def build_examples(task, split):
    """Return list of (input_text, target_text) in T5 text-to-text format."""
    from datasets import load_dataset
    ex = []
    if task == "sst2":
        ds = load_dataset("nyu-mll/glue", "sst2", split=split)
        for r in ds:
            if r["label"] < 0:
                continue
            ex.append((f"sst2 sentence: {r['sentence']}", SST2_LAB[r["label"]]))
    elif task == "mnli":
        ds = load_dataset("nyu-mll/glue", "mnli", split=split)
        for r in ds:
            if r["label"] < 0:
                continue
            ex.append((f"mnli premise: {r['premise']} hypothesis: {r['hypothesis']}", MNLI_LAB[r["label"]]))
    elif task == "race":
        ds = load_dataset("ehovy/race", "all", split=split)
        for r in ds:
            opts = r["options"]
            ai = "ABCD".index(r["answer"])
            o = " ".join(f"{c}: {t}" for c, t in zip("ABCD", opts))
            ex.append((f"question: {r['question']} options: {o} article: {r['article']}", opts[ai]))
    else:
        raise ValueError(task)
    return ex


def main() -> None:
    from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    assert not getattr(T5Config.from_pretrained(MODEL), "is_gated_act", False), \
        "use non-gated T5 v1.0 (t5-large), not t5-v1_1/flan"
    split = "train"
    ex = build_examples(TASK, split)
    if MAXTRAIN:
        ex = ex[:MAXTRAIN]
    print(f"  TASK={TASK} train_examples={len(ex)} SEQ={SEQ} batch={BS}x{ACCUM}={BS*ACCUM} lr={LR} epochs={EPOCHS}", flush=True)

    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL, dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt: torch.optim.Optimizer
    if OPT == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(model.parameters(), lr=LR, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR)
    print(f"  optimizer={OPT} lr={LR}", flush=True)
    rng = random.Random(0)

    steps_per_epoch = len(ex) // BS
    total_opt_steps = (steps_per_epoch * EPOCHS) // ACCUM
    step = 0; micro = 0; opt.zero_grad()
    for epoch in range(EPOCHS):
        order = list(range(len(ex))); rng.shuffle(order)
        for b in range(0, len(ex) - BS + 1, BS):
            idx = order[b:b + BS]
            src = [ex[i][0] for i in idx]; tgt = [ex[i][1] for i in idx]
            enc = tok(src, return_tensors="pt", padding=True, truncation=True, max_length=SEQ)
            lab = tok(tgt, return_tensors="pt", padding=True, truncation=True, max_length=16).input_ids
            lab[lab == tok.pad_token_id] = -100
            out = model(input_ids=enc.input_ids.to(dev), attention_mask=enc.attention_mask.to(dev),
                        labels=lab.to(dev))
            (out.loss / ACCUM).backward()
            micro += 1
            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % 50 == 0:
                    print(f"  epoch {epoch} step {step}/{total_opt_steps} loss {out.loss.item():.4f}", flush=True)

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"  SAVED fine-tuned {MODEL} on {TASK} to {OUT}", flush=True)


if __name__ == "__main__":
    main()
