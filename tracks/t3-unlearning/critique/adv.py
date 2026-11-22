"""Sélecteur « plus petite perturbation adversariale » (L-inf, PGD non ciblé) + classe atteinte."""
import numpy as np, torch, timm, pandas as pd
from scipy.stats import spearmanr
D="cuda"; torch.manual_seed(0)
m=timm.create_model("vit_base_patch16_224",pretrained=False,num_classes=100,init_values=1e-4)
m.load_state_dict(torch.load("data/model.pt",map_location="cpu",weights_only=True)); m=m.to(D).eval()
for p in m.parameters(): p.requires_grad_(False)
X=torch.from_numpy(np.load("data/forget_images.npz")["images"]); N=len(X)
old=pd.read_csv("sel.csv"); y=torch.tensor(old.orig_pred.values)
EPS=[0.25,0.5,1,2,4,8]  # en niveaux de gris /255
mineps=np.full(N,99.0); advcls=np.full(N,-1)
for i in range(0,N,100):
    x0=((X[i:i+100].to(D).permute(0,3,1,2).float()/255)-0.5)/0.5; yy=y[i:i+100].to(D)
    done=torch.zeros(len(x0),dtype=torch.bool,device=D)
    for e in EPS:
        eps=2*e/255; a=eps/3; d=torch.zeros_like(x0,requires_grad=True)
        for _ in range(7):
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=torch.nn.functional.cross_entropy(m(x0+d).float(),yy)
            g,=torch.autograd.grad(loss,d)
            d=(d+a*g.sign()).clamp(-eps,eps).detach().requires_grad_(True)
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.float16):
            pr=m((x0+d).clamp(-1,1)).argmax(1)
        new=(pr!=yy)&~done
        idx=new.nonzero().flatten().cpu().numpy()
        mineps[i+idx]=e; advcls[i+idx]=pr[new].cpu().numpy(); done|=new
    print(i,flush=True)
old["adv_eps"]=mineps; old["adv_class"]=advcls; old.to_csv("sel.csv",index=False,float_format="%.4f")
print("hist eps", {e:int((mineps==e).sum()) for e in EPS+[99.0]})
top=lambda v: set(np.argsort(v+1e-6*np.random.rand(N))[:700])
ha=top(mineps)
for c,s in [("margin",1),("pred_depth",-1),("oof_b8_loss",-1)]:
    print("spearman adv~",c, round(spearmanr(mineps,old[c]).correlation,3), "overlap top20", round(len(ha&top(s*old[c].values))/700,3))
w=advcls>=0
print("adv class == orig second", (advcls[w]==old.orig_second.values[w]).mean())
ow=old.oof_b8_pred.values!=old.orig_pred.values
print("adv class == oof wrong class (oof wrong)", (advcls[w&ow]==old.oof_b8_pred.values[w&ow]).mean(), (w&ow).sum())
