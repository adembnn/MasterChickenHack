python assemble.py --n 12 --no-rect --no-snap --no-ens --only 'reel/mixr|reel/pred|rang|pixelseal/pred|pixelseal/G' --skip 'rang2|residus' --boxes 't1-assemblage/boxtm2/|t1-assemblage/boxnet2/|t1-pixelseal/box_' > assemble_12.out 2> assemble_12.log
python compose.py 13 pixelseal=6 trustmark=12 stegastamp=12 ref=11 > compose_13.out 2>&1
python - > toutun_13.out 2>&1 <<'PY'
import torch, sys
sys.path.insert(0, "."); from t1lib import *
for n in (11, 13):
    s = torch.load(f"submission_candidat_{n}.pt", weights_only=True); meth = {r["image_name"]: r["method"] for r in load_test()}; fr = s["masks"].float().mean((1, 2))
    for m in METHODS:
        k = [i for i, nm in enumerate(s["image_names"]) if meth[nm] == m]; f = fr[k]; print(f"candidat {n} {m:10s} test : images tout-1 {int((f == 1).sum())}, tout-0 {int((f == 0).sum())} sur {len(k)}")
PY
touch assemble_12.fini
