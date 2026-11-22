"""Les formes (taches propres des pleines, zones filigranees des partielles) sont-elles tirees d'une bibliotheque finie ? Si la meme forme revient
d'une image a l'autre, les formes du train serviraient de dictionnaire pour le test (on accrocherait la carte floue a la forme connue la plus proche).
Mesure : pour chaque masque du train, meilleur IoU avec le masque d'une AUTRE image (tel quel, puis a translation pres par correlation)."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
R = load_train()
for nom, S in (("taches propres des pleines", [~r["gt"] for r in R if r["full"]]), ("zones filigranees des partielles", [r["gt"] for r in R if not r["full"]])):
    M = torch.from_numpy(np.stack(S)).float(); n = len(M); a = M.sum((1, 2)); flat = M.reshape(n, -1)
    inter = flat @ flat.T; iou = inter / (a[:, None] + a[None] - inter).clamp(min=1); iou.fill_diagonal_(0)
    best = iou.max(1).values.numpy()
    # a translation pres : correlation croisee (FFT), le meilleur recouvrement sur tous les decalages
    Fm = torch.fft.rfft2(F.pad(M, (0, 128, 0, 128))); bt = np.zeros(n)
    for i in range(n):
        c = torch.fft.irfft2(Fm[i:i + 1].conj() * Fm, s=(256, 256)).reshape(n, -1).max(1).values; u = (a[i] + a - c).clamp(min=1); v = c / u; v[i] = 0; bt[i] = float(v.max())
    print(f"{nom} (n={n}) : meilleur IoU avec une autre image, tel quel : mediane {np.median(best):.2f}, 90 % {np.quantile(best, .9):.2f}, max {best.max():.2f}, part > 0,8 : {(best > .8).mean():.2f}"
          f" | a translation pres : mediane {np.median(bt):.2f}, 90 % {np.quantile(bt, .9):.2f}, max {bt.max():.2f}, part > 0,8 : {(bt > .8).mean():.2f}")
