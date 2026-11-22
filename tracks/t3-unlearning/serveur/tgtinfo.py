import torch, glob, os
for f in sorted(glob.glob(os.path.expanduser("~/eac/t3/t3-methodes/cache/tgt_T85_f*_l*.pt"))):
    d = torch.load(f, map_location="cpu")
    a = d.get("args") if isinstance(d, dict) else None
    print(os.path.basename(f), list(d.keys())[:8] if isinstance(d, dict) else type(d), a if a else "")
