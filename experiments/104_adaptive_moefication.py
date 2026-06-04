import sys, pathlib, statistics as st
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
c=torch.load(pathlib.Path(__file__).resolve().parents[1]/"olmoe_layer0_cache.pt",weights_only=True)
W1,W2,W3=c["experts_W1"].float(),c["experts_W2"].float(),c["experts_W3"].float()
Wg=c["router_weight"].float(); H=c["H"].float(); N,K=c["n_experts"],c["top_k"]; dff=W1.shape[1]
topi=(H@Wg.T).topk(K,-1).indices
inset=torch.zeros(H.shape[0],N,dtype=torch.bool); inset.scatter_(1,topi,True)
G=64                      # groups (experts) of dff/G=16 units
def kmeans(X,k,iters=25):  # X:[n,d] rows=units
    c=X[torch.randperm(X.shape[0])[:k]].clone()
    for _ in range(iters):
        d=torch.cdist(X,c); a=d.argmin(1)
        for j in range(k):
            m=(a==j)
            if m.any(): c[j]=X[m].mean(0)
    return a
def err_curve(e, tokens):
    # build per-unit activation profile over tokens -> cluster units into G groups
    A=[]
    for t in tokens:
        A.append((F.silu(W1[e]@H[t])*(W2[e]@H[t])).abs())
    A=torch.stack(A,1)                 # [dff, n_tok]
    prof=A/ (A.norm(dim=1,keepdim=True)+1e-8)
    grp=kmeans(prof,G)                 # unit->group
    res={}
    for frac in [0.25,0.5]:
        Kg=max(1,int(frac*G)); errs_ad=[]; errs_st=[]
        for t in tokens:
            u=F.silu(W1[e]@H[t])*(W2[e]@H[t]); y=W3[e]@u; cn=u.abs()*W3[e].norm(dim=0)
            # adaptive: top-Kg groups by group contribution
            gc=torch.zeros(G)
            for g in range(G): gc[g]=cn[grp==g].sum()
            keepg=set(gc.topk(Kg).indices.tolist())
            mask=torch.tensor([1.0 if grp[k].item() in keepg else 0.0 for k in range(dff)])
            errs_ad.append(((W3[e]@(u*mask))-y).norm().item()/y.norm().item())
            # static: top (frac*dff) units globally
            ks=cn.topk(int(frac*dff)).indices; m2=torch.zeros(dff); m2[ks]=1
            errs_st.append(((W3[e]@(u*m2))-y).norm().item()/y.norm().item())
        res[frac]=(st.median(errs_ad),st.median(errs_st))
    return res
experts=[e for e in range(N) if inset[:,e].sum()>=5][:6]
agg={0.25:([],[]),0.5:([],[])}
for e in experts:
    toks=torch.nonzero(inset[:,e]).flatten().tolist()
    r=err_curve(e,toks)
    for f in r: agg[f][0].append(r[f][0]); agg[f][1].append(r[f][1])
print(f"SwiGLU FFN MoEfication: {dff} units -> {G} groups(experts), oracle top-Kg routing")
print(f"{'keep%':>6s} | {'ADAPTIVE-group err':>18s} | {'STATIC top-unit err':>19s}")
for f in [0.25,0.5]:
    print(f"{int(f*100):>5d}% | {st.median(agg[f][0])*100:>17.1f}% | {st.median(agg[f][1])*100:>18.1f}%")
print("ADAPTIVE << STATIC => co-activation grouping + routing escapes; ~equal => cancellation dominates.")
