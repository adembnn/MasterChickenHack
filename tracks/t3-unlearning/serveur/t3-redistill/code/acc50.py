import torch, numpy as np, torch.nn.functional as F
C = "/home/adem/eac/t3/t3-methodes/cache/"; R = "/home/adem/eac/t3/t3-redistill/tm/cache/"  # memes chemins sur h200c
o = torch.load(C + "oof_rd.pt"); y = o["yhat"]; cf = o["cls_fausse"]; N = len(y); ar = torch.arange(N); fc = y.unique()
lg0 = torch.from_numpy(np.load(C + "exact_orig_logits.npy")); o2 = lg0.clone(); o2[ar, y] = -1e9; second = o2.argmax(1)
for f in ["50", "60"]:
    b = torch.load(R + f"tgt_T85_f{f}_l814.pt"); fl = b["flip_idx"]; bc = b["z"][fl].argmax(1)
    print(f"T85_f{f}_l814 : {len(fl)} images basculees ; classe fausse redistill = classe cible de t3-methodes : {(cf[fl] == bc).float().mean():.3f} ; redistill = 2e classe d orig : {(cf[fl] == second[fl]).float().mean():.3f} ; t3-methodes = 2e classe d orig : {(bc == second[fl]).float().mean():.3f} ; les trois d accord : {((cf[fl] == bc) & (bc == second[fl])).float().mean():.3f}")
    print(f"   images de ce lot fausses hors pli : {o['wrong'][fl].float().mean():.3f} ; classe redistill parmi les 55 classes absentes : {1 - torch.isin(cf[fl], fc).float().mean():.3f} (t3-methodes : {1 - torch.isin(bc, fc).float().mean():.3f}) ; classes fausses distinctes : redistill {len(cf[fl].unique())}, t3-methodes {len(bc.unique())}")
    for n in [f"RDc{f}_l814", f"RDx{f}_l814"] + (["RDh50_l814"] if f == "50" else []):
        d = torch.load(R + f"tgt_{n}.pt"); g = d["flip_idx"]; com = torch.tensor(sorted(set(g.tolist()) & set(fl.tolist()))); ce = F.cross_entropy(d["z"], y, reduction="none")
        print(f"   {n} : {len(g)} basculees, {len(com)} communes avec T85_f{f} ; classe cible = classe redistill {(d['z'][g].argmax(1) == cf[g]).float().mean():.3f} ; = classe t3-methodes sur les communes {(d['z'][com].argmax(1) == b['z'][com].argmax(1)).float().mean():.3f} ; loss cible {ce.mean():.2f} / {ce.std():.2f}")
