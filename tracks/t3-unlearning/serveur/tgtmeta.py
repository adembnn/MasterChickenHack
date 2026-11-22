import torch, os, warnings; warnings.filterwarnings("ignore")
d = torch.load(os.path.expanduser("~/eac/t3/t3-methodes/cache/tgt_T85_f20_l1218.pt"), map_location="cpu")
print(d["meta"])
