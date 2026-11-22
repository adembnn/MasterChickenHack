"""Score brut (seuil 0) de toutes les cartes deposees par t1-reel et t1-pixelseal : reserve, hors reserve, pleines, partielles."""
import glob, sys, numpy as np
sys.path.insert(0, "."); from t1lib import *
R = {m: {r["image_name"]: r for r in load_train(m)} for m in METHODS}
for f in sorted(glob.glob(str(SESS / "t1-reel" / "*" / "pred_train_*.npz")) + glob.glob(str(SESS / "t1-reel" / "pred_train_*.npz")) + glob.glob(str(SESS / "t1-pixelseal" / "pred_train_*.npz")) + glob.glob("/tmp/eac-t1-partage/pred_train_*.npz")):
    try:
        z = np.load(f); n = [str(k) for k in z["names"]]; L = z["logits"].astype(np.float32)
        for m in METHODS:
            k = [i for i, x in enumerate(n) if x in R[m]]
            if len(k) < 100: continue
            rows = [R[m][n[i]] for i in k]; rep = report(miou(L[k] > 0, np.stack([r["gt"] for r in rows])), rows); import os, time
            print(f"{m:10s} {f.split('/eac/t1/')[-1][:58]:58s} n={len(k)} tous {rep['tous']:.4f} | reserve {rep['reserve']:.4f} | hors reserve {rep['hors_reserve']:.4f} | pleines {rep['pleines']:.4f} | partielles {rep['partielles']:.4f} | {time.strftime('%H:%M', time.localtime(os.path.getmtime(f)))} | test {'oui' if os.path.exists(f.replace('pred_train_', 'pred_test_')) else 'NON'}")
    except Exception as e: print(f, "ERREUR", str(e)[:80])
