import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
c = torch.load(pathlib.Path(__file__).resolve().parents[1]/"olmoe_layer0_cache.pt", weights_only=True)
W1,W2,W3 = c["experts_W1"].float(),c["experts_W2"].float(),c["experts_W3"].float()
Wg=c["router_weight"].float(); H=c["H"].float(); N,K=c["n_experts"],c["top_k"]
sw=lambda h,a,b,d:(F.silu(h@a.T)*(h@b.T))@d.T
topv,topi=(H@Wg.T).topk(K,-1); g=torch.softmax(topv,-1)
ratios=[]; droprel=[]; usedcnt=set()
for t in range(H.shape[0]):
    gE=[]; 
    for j in range(K):
        e=topi[t,j].item(); usedcnt.add(e)
        gE.append(g[t,j]*sw(H[t:t+1],W1[e],W2[e],W3[e]).squeeze(0))
    gE=torch.stack(gE)              # [K,d]
    Y=gE.sum(0)                     # layer output (gated sum)
    sum_norms=gE.norm(dim=1).sum()  # Σ ||g_e E_e||
    ratios.append((Y.norm()/sum_norms).item())
    # drop the smallest-contribution active expert -> relative output change
    s=gE.norm(dim=1).argmin()
    droprel.append((gE[s].norm()/Y.norm()).item())
import statistics as st
print(f"tokens={H.shape[0]}, distinct experts used (1 layer)= {len(usedcnt)}/{N}")
print(f"cancellation ratio  ||Y|| / Σ||g_e E_e||  : median {st.median(ratios):.3f}  (1=no cancel, →0 = full cancel)")
print(f"=> active experts' contributions sum to ~{1/st.median(ratios):.1f}x the output norm (cancel ~{(1-st.median(ratios))*100:.0f}%)")
print(f"drop the SMALLEST active expert: ΔY/||Y|| median {st.median(droprel)*100:.0f}%  (relative output change from removing the *least* important active expert)")
