import sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
c=torch.load(pathlib.Path(__file__).resolve().parents[1]/"olmoe_layer0_cache.pt",weights_only=True)
W1,W2,W3=c["experts_W1"].float(),c["experts_W2"].float(),c["experts_W3"].float()
Wg=c["router_weight"].float(); H=c["H"].float(); N,K=c["n_experts"],c["top_k"]; dff=W1.shape[1]
topi=(H@Wg.T).topk(K,-1).indices
inset=torch.zeros(H.shape[0],N,dtype=torch.bool); inset.scatter_(1,topi,True)
experts=[e for e in range(N) if inset[:,e].sum()>=6][:8]
for frac in [0.25,0.5]:
    plain=[]; biasc=[]; lowrank=[]
    for e in experts:
        toks=torch.nonzero(inset[:,e]).flatten().tolist()
        Ys=[]; Ks=[]; Ds=[]
        for t in toks:
            u=F.silu(W1[e]@H[t])*(W2[e]@H[t]); y=W3[e]@u; cn=u.abs()*W3[e].norm(dim=0)
            keep=cn.topk(int(frac*dff)).indices; m=torch.zeros(dff); m[keep]=1
            yk=W3[e]@(u*m)
            Ys.append(y); Ks.append(yk); Ds.append(y-yk)   # dropped contribution
        Y=torch.stack(Ys); Kp=torch.stack(Ks); D=torch.stack(Ds)
        # plain drop error
        plain += (((Kp-Y).norm(dim=1))/(Y.norm(dim=1))).tolist()
        # bias correction = mean dropped contribution
        b=D.mean(0)
        biasc += (((Kp+b-Y).norm(dim=1))/(Y.norm(dim=1))).tolist()
        # rank-r correction of the dropped part (best rank-4 over these tokens)
        U,S,Vt=torch.linalg.svd(D-b, full_matrices=False); r=min(4,S.shape[0])
        Dr=b+ (U[:,:r]*S[:r])@Vt[:r]
        lowrank += (((Kp+Dr-Y).norm(dim=1))/(Y.norm(dim=1))).tolist()
    print(f"keep {int(frac*100)}% units: plain-drop {st.median(plain)*100:5.1f}% | "
          f"+mean-bias {st.median(biasc)*100:5.1f}% | +rank-4 corr {st.median(lowrank)*100:5.1f}%")
print("if +correction << plain-drop, the dropped (cancelling) part is predictable -> shared-base/residual MoEfication beats pure removal.")
