"""T1 · régénère les prédictions test d'un filigrane à partir des poids sauvés (moyenne des plis et des variantes).
Usage : python predict.py --method trustmark --tags p_mixr+p_dec [--tta 2] [--train 1] [--out DOSSIER]
Écrit pred_test_<methode>.npz (clés names, logits float16 N x 128 x 128, > 0 = filigrané) ; avec --train 1, vérifie aussi que le score de réserve
recalculé depuis les poids est le même que celui des prédictions sauvées à l'entraînement."""
import sys, json, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel7 as R

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--tags", required=True); ap.add_argument("--tta", type=int, default=2)
ap.add_argument("--train", type=int, default=0); ap.add_argument("--out", default=""); a = ap.parse_args()
torch.set_num_threads(R.NT); dev = "cuda"; trm, tem = R.metas(); method = a.method
RE = np.load(R.CACHE / "REAL_test.npy", mmap_mode="r"); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); Y128 = np.load(R.CACHE / "Y128_train.npy")
own_te = [i for i, m in enumerate(tem) if m["method"] == method]; idx, res, rest, fold = R.splits(trm, method); pos = {i: p for p, i in enumerate(idx)}
XT = np.ascontiguousarray(RE[own_te]); XR = np.ascontiguousarray(RT[res]); test, resv, n = 0, 0, 0
for tag in a.tags.split("+"):
    for f in range(5):
        ck = torch.load(R.ROOT / "runs" / tag / method / f"fold{f}" / "ema.pt", map_location=dev); args = argparse.Namespace(**ck["args"]); args.pretrained = 0
        for k_, v_ in (('s1', 0), ('st', 0), ('echan', 0), ('hp', 0), ('flag', 0), ('stem1', 0), ('aux', 0.0)): setattr(args, k_, getattr(args, k_, v_))   # options absentes des anciens poids
        assert not args.echan, 'predict.py ne gère pas le canal de force attendue (variante abandonnée)'
        D = R.ROOT / f"data_{method}{args.suffix.split(',')[0]}"; FT = (np.load(D / "AALL_test.npy") > 0.75).astype(np.float32); FR = (np.load(D / "AALL_train.npy")[[pos[i] for i in res]] > 0.75).astype(np.float32)
        run = R.Runner(args, dev); print(tag, f, run.net.load_state_dict(ck["state"]), flush=True)
        test = test + run.predict(run.net, XT, FT, tta=a.tta); n += 1
        if a.train: resv = resv + run.predict(run.net, XR, FR, tta=a.tta)
        del run; torch.cuda.empty_cache()
out = Path(a.out) if a.out else R.ROOT / "runs" / a.tags / method; out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / f"pred_test_{method}.npz", names=np.array([tem[i]["image_name"] for i in own_te]), logits=(test / n).astype(np.float16))
print("écrit", out / f"pred_test_{method}.npz", "| part prédite filigranée (seuil 0) :", round(float((test / n > 0).mean()), 3))
if a.train: print("réserve recalculée depuis les poids, mIoU seuil 0 :", round(float(R.miou_list(resv / n > 0, Y128[res]).mean()), 4))
