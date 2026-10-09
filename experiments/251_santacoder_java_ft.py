"""Experiment 251 — fine-tune base SantaCoder-1.1B on Java code, to match G-MoEfication's
setup (they fine-tune SantaCoder on the MegaCodeTraining set filtered to Java only; their
fine-tuned dense MultiPL-E Java pass@1 = 17.81%, vs base ~15.2%). lr 1e-5, batch 8 (their
recipe). Java code = the ```java ... ``` blocks in the ASSISTANT field. Saves a checkpoint
for 243 to evaluate (HHMODEL=<out>). Run: python3 experiments/251_santacoder_java_ft.py
"""
from __future__ import annotations
import os, re, math, pathlib
import torch, torch.nn.functional as F

MODEL = os.environ.get("HHMODEL", "bigcode/gpt_bigcode-santacoder")
OUT = os.environ.get("OUT", str(pathlib.Path(__file__).resolve().parents[1] / "ckpts" / "santacoder-java"))
SEQ = int(os.environ.get("SEQ", "512"))
BS = int(os.environ.get("BS", "2"))
ACCUM = int(os.environ.get("ACCUM", "4"))      # effective batch = BS*ACCUM (paper: 8)
LR = float(os.environ.get("LR", "1e-5"))
STEPS = int(os.environ.get("STEPS", "2500"))   # optimizer steps
JAVA_RE = re.compile(r"```java\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- collect Java code blocks from MegaCodeTraining ASSISTANT field ---
    ds = load_dataset("rombodawg/MegaCodeTraining", split="train")
    chunks = []
    njava = 0
    for ex in ds:
        a = ex.get("ASSISTANT") or ""
        blocks = JAVA_RE.findall(a)
        if not blocks:
            continue
        njava += 1
        for b in blocks:
            b = b.strip()
            if len(b) > 40:
                chunks.append(b)
    print(f"  Java examples: {njava}, code blocks: {len(chunks)}", flush=True)
    text = "\n\n".join(chunks)
    ids = tok(text, return_tensors="pt").input_ids[0]
    nblk = ids.numel() // SEQ
    ids = ids[:nblk * SEQ].view(nblk, SEQ)
    print(f"  tokens: {nblk*SEQ}, blocks of {SEQ}: {nblk}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(0)

    step = 0; micro = 0; opt.zero_grad()
    done = False
    while not done:
        perm = torch.randperm(nblk, generator=gen)
        for i in range(0, nblk - BS + 1, BS):
            x = ids[perm[i:i + BS]].to(dev)
            loss = model(input_ids=x, labels=x).loss / ACCUM
            loss.backward()
            micro += 1
            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"  step {step}/{STEPS} loss {loss.item()*ACCUM:.4f}", flush=True)
                if step >= STEPS:
                    done = True; break

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"  SAVED fine-tuned SantaCoder-Java to {OUT}", flush=True)


if __name__ == "__main__":
    main()
