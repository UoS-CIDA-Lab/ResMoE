import sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
c=torch.load(pathlib.Path(__file__).resolve().parents[1]/"olmoe_layer0_cache.pt",weights_only=True)
W1,W2,W3=c["experts_W1"].float(),c["experts_W2"].float(),c["experts_W3"].float()
Wg=c["router_weight"].float(); H=c["H"].float(); N,K=c["n_experts"],c["top_k"]; dff=W1.shape[1]
topi=(H@Wg.T).topk(K,-1).indices; inset=torch.zeros(H.shape[0],N,dtype=torch.bool); inset.scatter_(1,topi,True)
dens=[]; negfrac=[]; gate_neg=[]; relu_zero=[]
for e in range(N):
    for t in torch.nonzero(inset[:,e]).flatten()[:3].tolist():
        h=H[t]
        pre1=W1[e]@h; v=W2[e]@h; u=F.silu(pre1)*v       # SwiGLU hidden
        # density: fraction of units with |u| >= 1% of max|u|
        dens.append((u.abs()>=0.01*u.abs().max()).float().mean().item())
        negfrac.append((u<0).float().mean().item())       # signed coefficients
        gate_neg.append((v<0).float().mean().item())       # the W2h gate sign source
        # if this FFN were ReLU on the same pre-activation W1h: fraction exactly 0
        relu_zero.append((pre1<=0).float().mean().item())
print(f"SwiGLU hidden u = SiLU(W1h) * (W2h),  dff={dff}")
print(f"  ACTIVE density (|u|>=1% of max)      : {st.median(dens)*100:5.1f}%  (high = dense, nothing to skip)")
print(f"  SIGNED: fraction u_k < 0             : {st.median(negfrac)*100:5.1f}%  (~50% => heavy sign cancellation)")
print(f"   - driven by gate (W2h)<0 fraction   : {st.median(gate_neg)*100:5.1f}%")
print(f"  if ReLU(W1h) instead: fraction == 0  : {st.median(relu_zero)*100:5.1f}%  (high = hard sparsity, skippable)")
