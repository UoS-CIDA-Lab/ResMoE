"""Probe: Phi-2 structure + dense zero-shot SuperGLUE-subset sanity check.
Run: python3 experiments/116_phi2_probe.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

dev = "cuda"
tok = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", dtype=torch.float16, trust_remote_code=True).to(dev).eval()
model.config.use_cache = False
torch.set_grad_enabled(False)
L = model.model.layers
mlp = L[0].mlp
print("n_layers", len(L), "| d_model", model.config.hidden_size,
      "| intermediate", model.config.intermediate_size)
print("mlp type:", type(mlp).__name__, "| children:", [n for n, _ in mlp.named_children()])
print("fc1", tuple(mlp.fc1.weight.shape), "fc2", tuple(mlp.fc2.weight.shape),
      "act", type(mlp.activation_fn).__name__)


@torch.no_grad()
def ll(prompt, cont):
    pi = tok(prompt, return_tensors="pt").input_ids.to(dev)
    ci = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    ids = torch.cat([pi, ci], 1)
    lg = model(ids).logits[0].float().log_softmax(-1)
    n = ci.shape[1]
    return lg[-n - 1:-1].gather(1, ci[0].unsqueeze(1)).sum().item()


ds = load_dataset("super_glue", "boolq", split="validation", trust_remote_code=True)
N = 80
correct = 0
for ex in ds.select(range(N)):
    p = f"{ex['passage']}\nQuestion: {ex['question']}?\nAnswer:"
    correct += int((1 if ll(p, " yes") > ll(p, " no") else 0) == ex["label"])
print(f"BoolQ dense zero-shot acc ({N}): {correct / N:.3f}  (chance 0.5, Phi-2 ~0.83)")
