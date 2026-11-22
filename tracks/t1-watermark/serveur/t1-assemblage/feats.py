"""Signaux "image entiere" pour reconnaitre la population (pleine / partielle) sans la verite : marge et taux de bits justes du decodeur
sur l'image entiere, ses 4 quarts et son centre. Sortie : feat_<split>.npz (names, method, acc, margin, qacc[N,5], qmargin[N,5])."""
import sys, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
torch.set_num_threads(4)
for split in ("train", "test"):
    N, M, A, MG, QA, QM = [], [], [], [], [], []; t0 = time.time()
    for method in METHODS:
        wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
        rows = load_train(method) if split == "train" else load_test(method)
        for r in rows:
            x = to_tensor(load_img(split, r["image_name"])); sgn = torch.tensor([int(c) for c in r["message"]]).float() * 2 - 1
            mc = x.mean(dim=(2, 3), keepdim=True)
            b = [x]
            for (y, xx) in ((0, 0), (0, 256), (256, 0), (256, 256), (128, 128)):       # on ne garde qu'un quart, le reste = couleur moyenne
                t = mc.expand(1, 3, 512, 512).clone(); t[:, :, y:y + 256, xx:xx + 256] = x[:, :, y:y + 256, xx:xx + 256]; b.append(t)
            with torch.no_grad(): lg = wm.logits(torch.cat(b).cuda() if method != "stegastamp" else torch.cat(b)).cpu().float()
            mg = (lg * sgn).mean(1).numpy(); ac = (((lg > 0).float() * 2 - 1) == sgn).float().mean(1).numpy()
            N.append(r["image_name"]); M.append(method); A.append(ac[0]); MG.append(mg[0]); QA.append(ac[1:]); QM.append(mg[1:])
        del wm; torch.cuda.empty_cache()
        print(split, method, len(rows), round(time.time() - t0), "s", flush=True)
    np.savez(ME / f"feat_{split}.npz", names=np.array(N), method=np.array(M), acc=np.array(A, np.float32), margin=np.array(MG, np.float32), qacc=np.array(QA, np.float32), qmargin=np.array(QM, np.float32))
# controle sur le train : la precision de bits separe-t-elle les deux populations ?
z = np.load(ME / "feat_train.npz"); rows = {r["image_name"]: r for r in load_train()}
for method in METHODS:
    k = z["method"] == method; full = np.array([rows[n]["full"] for n in z["names"][k]]); acc = z["acc"][k]
    ths = np.unique(acc); best = max(((np.mean((acc > t) == full), t) for t in ths))
    print("POP", method, "bits justes pleines", np.quantile(acc[full], [0, .1, .5, .9, 1]).round(3), "partielles", np.quantile(acc[~full], [0, .1, .5, .9, 1]).round(3), "meilleur seuil", round(float(best[1]), 3), "exactitude", round(float(best[0]), 3))
