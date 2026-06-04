import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F, statistics as st
c=torch.load(pathlib.Path(__file__).resolve().parents[1]/"olmoe_layer0_cache.pt",weights_only=True)
W1,W2,W3=c["experts_W1"].float(),c["experts_W2"].float(),c["experts_W3"].float()
Wg=c["router_weight"].float(); H=c["H"].float(); N,K=c["n_experts"],c["top_k"]; dff=W1.shape[1]
topi=(H@Wg.T).topk(K,-1).indices
inset=torch.zeros(H.shape[0],N,dtype=torch.bool); inset.scatter_(1,topi,True)
# treat each OLMoE expert as a dense SwiGLU FFN (dff=1024 hidden units); measure unit-level slack
cancel=[]; eff90=[]; keep_err={r:[] for r in [0.1,0.25,0.5]}
for e in range(N):
    idx=torch.nonzero(inset[:,e]).flatten()
    for t in idx[:3].tolist():
        h=H[t]
        u=F.silu(W1[e]@h)*(W2[e]@h)            # [dff] hidden activations
        contrib=u.abs()*W3[e].norm(dim=0)      # per-unit contribution magnitude
        y=W3[e]@u                              # FFN output
        # cancellation at unit level: ||y|| / sum_k ||contrib_k||
        cancel.append((y.norm()/contrib.sum()).item())
        order=contrib.argsort(descending=True)
        # effective units for 90% of output energy (keep top-r, measure recon)
        for r in keep_err:
            keep=order[:int(r*dff)]
            mask=torch.zeros(dff); mask[keep]=1
            yk=W3[e]@(u*mask)
            keep_err[r].append((yk-y).norm().item()/y.norm().item())
        # effective rank: smallest r s.t. recon error<=0.1
        cum=None
        for rr in range(1,dff+1):
            pass
print(f"FFN hidden units per expert: {dff}")
print(f"unit-level cancellation ||y||/Σ||c_k|| : median {st.median(cancel):.3f}  (lower=more cancel; MoE expert-level was 0.40)")
for r in sorted(keep_err):
    print(f"keep top {int(r*100)}% units ({int(r*dff)}/{dff}): output error median {st.median(keep_err[r])*100:5.1f}%")
